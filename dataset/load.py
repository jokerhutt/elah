from itertools import islice

from config import DATASETS, IM_END, IM_START

from .parse import parse_file


TOKENIZER_AMOUNTS = {
    "fineweb_edu": 45_000,

    "tinystories": 10_000,

    "codeparrot": 15_000,

    "cosmopedia_auto_math": 5_000,
    "cosmopedia_khanacademy": 3_000,
    "cosmopedia_openstax": 3_000,
    "cosmopedia_stanford": 8_000,
    "cosmopedia_stories": 8_000,
    "cosmopedia_wikihow": 3_000,

    "open_web_math": 10_000,

    "tulu3": 3_000,

}

def iter_tokenizer_data():
    for name, amount in TOKENIZER_AMOUNTS.items():
        for sample in islice(_iter_dataset(name), amount):
            yield sample if isinstance(sample, str) else format_chat(sample)


def format_chat(messages):
    return "\n".join(
        f"{IM_START}{m['role']}\n{m['content']}{IM_END}"
        for m in messages
        if m.get("content")
    )


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