import json
import logging
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()


def get_logger(name: str = "elah"):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(
            RichHandler(
                console=console,
                markup=True,
                rich_tracebacks=True,
                show_path=False,
                log_time_format="[%H:%M:%S]",
            )
        )

    return logger


def log_panel(body: str, title: str):
    console.print(Panel(body, title=title, border_style="cyan"))


def log_settings(title: str, sections: dict[str, dict]):
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column(style="bold")

    for index, (heading, fields) in enumerate(sections.items()):
        if index:
            table.add_row("", "")

        table.add_row("", f"[cyan]{heading}[/]")

        for key, value in fields.items():
            table.add_row(key, str(value))

    console.print(Panel(table, title=title, border_style="cyan"))


def format_duration(seconds: float):
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h{minutes:02d}m"

    if minutes:
        return f"{minutes}m{seconds:02d}s"

    return f"{seconds}s"


class MetricsLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()

    def write(self, **fields):
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "elapsed": round(time.monotonic() - self.started, 1),
            **fields,
        }

        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return record


class NullProgress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def add_task(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        pass


def training_progress(enabled: bool = True):
    if not enabled:
        return NullProgress()

    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("loss [green]{task.fields[loss]}[/]"),
        TextColumn("gnorm [yellow]{task.fields[grad_norm]}[/]"),
        TextColumn("[red]{task.fields[skipped]}[/]"),
        TextColumn("elapsed"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(compact=True),
        console=console,
        refresh_per_second=4,
    )
