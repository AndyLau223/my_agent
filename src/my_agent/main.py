import uuid
import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage
from my_agent.graph.graph import get_graph

app = typer.Typer(help="My Agent — hierarchical multi-agent system CLI")
console = Console()


@app.command()
def run(
    task: str = typer.Argument(..., help="The task for the agents to complete."),
    thread_id: str = typer.Option(
        None, "--thread-id", "-t", help="Resume an existing run by thread ID."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print all agent messages."),
):
    """Submit a task to the multi-agent system and stream results."""
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    console.print(f"\n[bold cyan]Thread ID:[/bold cyan] {thread_id}")
    console.print(f"[bold green]Task:[/bold green] {task}\n")

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [HumanMessage(content=task)],
        "plan": [],
        "current_step": 0,
        "critique": "",
        "iteration": 0,
        "final_output": "",
    }

    try:
        for event in graph.stream(initial_state, config=config, stream_mode="values"):
            if verbose:
                messages = event.get("messages", [])
                if messages:
                    last = messages[-1]
                    console.print(f"[dim]{type(last).__name__}:[/dim] {last.content[:300]}")

            final = event.get("final_output", "")
            if final:
                console.print(
                    Panel(
                        Markdown(final),
                        title="[bold green]✓ Final Output[/bold green]",
                        border_style="green",
                    )
                )
                return

        console.print("[yellow]Run completed (no final output captured).[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/red]")
        raise typer.Exit(code=1)


@app.command()
def history(
    thread_id: str = typer.Argument(..., help="Thread ID to retrieve history for."),
):
    """Print the message history for a given thread."""
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
        if not messages:
            console.print("[yellow]No messages found for this thread.[/yellow]")
            return

        console.print(f"\n[bold]History for thread[/bold] {thread_id}:\n")
        for i, msg in enumerate(messages, 1):
            kind = type(msg).__name__
            content = msg.content[:500] if msg.content else "(empty)"
            console.print(f"[bold]{i}. {kind}[/bold]\n{content}\n")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
