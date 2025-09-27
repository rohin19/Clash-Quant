from __future__ import annotations

from rich.console import Console
from rich.table import Table
from datetime import datetime

console = Console()

def info(msg: str):
    console.log(f"[bold cyan][INFO][/bold cyan] {msg}")

def warn(msg: str):
    console.log(f"[bold yellow][WARN][/bold yellow] {msg}")

def error(msg: str):
    console.log(f"[bold red][ERROR][/bold red] {msg}")

def banner(title: str):
    console.rule(f"[bold green]{title}")

def table_dict(title: str, mapping: dict[str, int]):
    table = Table(title=title)
    table.add_column("Key")
    table.add_column("Value")
    for k, v in mapping.items():
        table.add_row(str(k), str(v))
    console.print(table)

def timestamp() -> str:
    return datetime.utcnow().isoformat()
