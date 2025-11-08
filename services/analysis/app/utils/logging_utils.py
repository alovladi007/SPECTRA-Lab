from rich.console import Console
console = Console()
def log_section(title:str):
    console.rule(f"[bold cyan]{title}[/]")
def info(msg:str):
    console.print(f"[bold green]✓[/] {msg}")
def warn(msg:str):
    console.print(f"[bold yellow]![/] {msg}")
def err(msg:str):
    console.print(f"[bold red]✗[/] {msg}")
