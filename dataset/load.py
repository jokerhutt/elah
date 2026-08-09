from config import DATASETS

from .parse import parse_file


def iter_pretraining_data():
    for name, config in DATASETS.items():
        if config.get("format") != "chat":
            yield from _iter_dataset(name)


def iter_sft_data():
    for name, config in DATASETS.items():
        if config.get("format") == "chat":
            yield from _iter_dataset(name)


def _iter_dataset(name):
    config = DATASETS[name]

    for filename in config["files"]:
        path = config["root"] / filename

        if not path.exists():
            raise FileNotFoundError(path)

        yield from parse_file(path, config)
