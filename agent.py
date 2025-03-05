from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import List, Dict
from tavily import TavilyClient
import os
import json
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.traceback import install
import time

# Install rich traceback handler
install()

# Initialize Rich console for better formatting
console = Console()

# Load environment variables
load_dotenv()


class Queries(BaseModel):
    """Model for search queries."""
    queries: List[str]


class EssayAgent:
    def __init__(self, model_name="gpt-4o-mini", temperature=0):
        """Initialize the essay agent with a model and memory."""
        self.model = ChatOpenAI(model=model_name, temperature=temperature)
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        # Add storage for tracking the latest draft
        self.latest_draft = None

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
        console.log("[bold cyan]🔍 Planning essay outline...[/bold cyan]")
        state["plan"] = self.query_model(self.prompts["plan"], state["task"])
        console.print(Panel(Markdown(state["plan"]), title="Essay Outline", border_style="green"))
        return state

    def research(self, state: Dict, prompt_key: str):
        """Conducts research using Tavily for additional content."""
        stage = "initial" if prompt_key == "research_plan" else "revision"
        console.log(f"[bold yellow]📚 Conducting {stage} research...[/bold yellow]")
        
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content=self.prompts[prompt_key]),
            HumanMessage(content=state["task"] if prompt_key == "research_plan" else state["critique"])
        ])
        
        console.print("Search queries:", style="bold")
        for i, q in enumerate(queries.queries, 1):
            console.print(f"  {i}. {q}")
            
        content = state.get("content", [])
        
        for q in queries.queries:
            console.log(f"Searching for: '{q}'")
            response = self.tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
                console.print(f"  - Found: {r['title'][:50]}...")

        state["content"] = content
        return state

    def generate_essay(self, state: Dict):
        """Generates the first draft or revised essay."""
        revision = state.get("revision_number", 1)
        console.log(f"[bold magenta]✍️ {'Writing' if revision == 1 else 'Revising'} essay (Version {revision})...[/bold magenta]")
        
        content = "\n\n".join(state.get("content", []))
        user_message = f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
        system_message = self.prompts["writer"].format(content=content)

        draft = self.query_model(system_message, user_message)
        state["draft"] = draft
        # Store the draft in the class instance
        self.latest_draft = draft
        
        state["revision_number"] = revision + 1
        
        console.print(Panel(Markdown(draft), title=f"Essay Draft v{revision}", border_style="blue"))
        return state

    def reflect_on_essay(self, state: Dict):
        """Generates critique and feedback on the essay."""
        console.log("[bold red]📝 Evaluating essay quality...[/bold red]")
        state["critique"] = self.query_model(self.prompts["reflection"], state["draft"])
        console.print(Panel(Markdown(state["critique"]), title="Essay Critique", border_style="yellow"))
        return state

    def should_continue(self, state: Dict):
        """Determines if the revision process should continue."""
        if state["revision_number"] > state["max_revisions"]:
            console.log("[bold green]✅ Reached maximum revisions. Finishing...[/bold green]")
            return END
        else:
            console.log("[bold]⟳ Continuing to next revision cycle...[/bold]")
            return "reflect"

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


def save_essay_to_file(text, topic):
    """Save essay to a file with error handling."""
    try:
        filename = f"essay_{topic.replace(' ', '_')[:20]}.md"
        with open(filename, "w") as f:
            f.write(text)
        return filename
    except Exception as e:
        console.print(f"[bold red]Error saving file: {str(e)}[/bold red]")
        return None


if __name__ == "__main__":
    try:
        console.print(Panel.fit("🤖 [bold]Essay Writing Agent[/bold]", style="cyan"))
        console.print("This agent writes essays with research and multiple revisions\n")
        
        topic = input("Enter essay topic: ") or "What is the difference between Langchain and Langsmith?"
        revisions = input("Max revision cycles (default: 2): ") or "2"
        
        console.print("\n[bold]Starting essay generation process...[/bold]\n")
        
        agent = EssayAgent()
        graph = agent.build_workflow()

        thread = {"configurable": {"thread_id": "1"}}
        
        # Process all state updates
        for _ in graph.stream({
            "task": topic,
            "max_revisions": int(revisions),
            "revision_number": 1,
        }, thread):
            # We don't need to do anything here as the agent tracks the latest draft
            pass
        
        # Add a small delay to ensure all console output has completed
        time.sleep(0.5)
        
        console.print("\n[bold green]✅ Essay generation complete![/bold green]")
        
        # Display the final essay using the agent's stored draft
        if agent.latest_draft:
            console.print("\nFinal Essay:", style="bold")
            console.print(Panel(Markdown(agent.latest_draft), title="Final Essay", border_style="green"))
            
            # Save the essay to a file
            filename = save_essay_to_file(agent.latest_draft, topic)
            if filename:
                console.print(f"\nEssay saved to [bold]{filename}[/bold]")
        else:
            console.print("[bold red]Warning: No essay draft was produced.[/bold red]")
            console.print("Check your API keys in the .env file and ensure both OpenAI and Tavily services are working.")
    
    except Exception as e:
        console.print_exception()
        console.print(f"\n[bold red]An error occurred: {str(e)}[/bold red]")