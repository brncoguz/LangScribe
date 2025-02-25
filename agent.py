from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel
from typing import List, Dict
from tavily import TavilyClient
import os

# Load environment variables
load_dotenv()


class Queries(BaseModel):
    queries: List[str]


class EssayAgent:
    def __init__(self, model_name="gpt-4o-mini", temperature=0):
        """Initialize the essay agent with a model and memory."""
        self.model = ChatOpenAI(model=model_name, temperature=temperature)
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

        self.prompts = {
            "plan": "You are an expert writer tasked with writing a high-level outline of an essay. "
                    "Write an outline for the user-provided topic with relevant notes and instructions.",

            "writer": "You are an essay assistant tasked with writing excellent 5-paragraph essays. "
                      "Generate the best essay possible for the user's request and the initial outline. "
                      "If the user provides critique, respond with a revised version of previous attempts. "
                      "Utilize all the information below as needed:\n\n------\n\n{content}",

            "reflection": "You are a teacher grading an essay submission. "
                          "Generate critique and recommendations for the user's submission, "
                          "including requests for length, depth, and style.",

            "research_plan": "You are a researcher gathering information for an essay. "
                             "Generate a list of search queries to find relevant information. "
                             "Only generate 3 queries max.",

            "research_critique": "You are a researcher refining an essay. "
                                 "Generate a list of search queries to improve requested revisions. "
                                 "Only generate 3 queries max."
        }

    def query_model(self, system_prompt: str, user_input: str):
        """Helper function to interact with the model."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]
        response = self.model.invoke(messages)
        return response.content

    def plan_essay(self, state: Dict):
        """Generates a high-level outline for the essay topic."""
        state["plan"] = self.query_model(self.prompts["plan"], state["task"])
        return state

    def research(self, state: Dict, prompt_key: str):
        """Conducts research using Tavily for additional content."""
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content=self.prompts[prompt_key]),
            HumanMessage(content=state["task"] if prompt_key == "research_plan" else state["critique"])
        ])
        content = state.get("content", [])

        for q in queries.queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])

        state["content"] = content
        return state

    def generate_essay(self, state: Dict):
        """Generates the first draft or revised essay."""
        content = "\n\n".join(state.get("content", []))
        user_message = f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
        system_message = self.prompts["writer"].format(content=content)

        state["draft"] = self.query_model(system_message, user_message)
        state["revision_number"] = state.get("revision_number", 1) + 1
        return state

    def reflect_on_essay(self, state: Dict):
        """Generates critique and feedback on the essay."""
        state["critique"] = self.query_model(self.prompts["reflection"], state["draft"])
        return state

    def should_continue(self, state: Dict):
        """Determines if the revision process should continue."""
        return END if state["revision_number"] > state["max_revisions"] else "reflect"

    def build_workflow(self):
        """Creates and compiles the Langgraph workflow."""
        builder = StateGraph(Dict)
        builder.add_node("planner", self.plan_essay)
        builder.add_node("generate", self.generate_essay)
        builder.add_node("reflect", self.reflect_on_essay)
        builder.add_node("research_plan", lambda s: self.research(s, "research_plan"))
        builder.add_node("research_critique", lambda s: self.research(s, "research_critique"))

        builder.set_entry_point("planner")

        builder.add_conditional_edges("generate", self.should_continue, {END: END, "reflect": "reflect"})
        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")
        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")

        return builder.compile()


if __name__ == "__main__":
    agent = EssayAgent()
    graph = agent.build_workflow()

    thread = {"configurable": {"thread_id": "1"}}
    for s in graph.stream({
        "task": "What is the difference between Langchain and Langsmith?",
        "max_revisions": 2,
        "revision_number": 1,
    }, thread):
        print(s)