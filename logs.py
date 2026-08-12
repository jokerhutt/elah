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
        TextColumn("elapsed"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(compact=True),
        console=console,
        refresh_per_second=4,
    )
