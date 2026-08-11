import random
from itertools import islice

from config import DATASETS, IM_END, IM_START

from .parse import parse_file


SHUFFLE_BUFFER = 10_000


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

MAX_SAMPLE_CHARS = 4_000

def iter_tokenizer_data():
    for name, amount in TOKENIZER_AMOUNTS.items():
        for sample in islice(_iter_dataset(name), amount):
            text = sample if isinstance(sample, str) else format_chat(sample)

            for start in range(0, len(text), MAX_SAMPLE_CHARS):
                chunk = text[start : start + MAX_SAMPLE_CHARS]

                if chunk.strip():
                    yield chunk


def format_chat(messages):
    return "\n".join(
        f"{IM_START}{m['role']}\n{m['content']}{IM_END}"
        for m in messages
        if m.get("content")
    )


def iter_pretraining_data():
    names = [name for name, config in DATASETS.items() if config.get("format") != "chat"]

    yield from _shuffled(_interleaved(names))


def iter_sft_data():
    names = [name for name, config in DATASETS.items() if config.get("format") == "chat"]

    yield from _shuffled(_interleaved(names))


def _interleaved(names):
    iterators = {name: _iter_dataset(name) for name in names}

    while iterators:
        for name in list(iterators):
            try:
                yield next(iterators[name])
            except StopIteration:
                del iterators[name]


def _shuffled(samples, buffer_size=SHUFFLE_BUFFER):
    buffer = []

    for sample in samples:
        buffer.append(sample)

        if len(buffer) >= buffer_size:
            i = random.randrange(len(buffer))
            buffer[i], buffer[-1] = buffer[-1], buffer[i]

            yield buffer.pop()

    random.shuffle(buffer)

    yield from buffer


def _iter_dataset(name):
    config = DATASETS[name]

    for filename in config["files"]:
        path = config["root"] / filename

        if not path.exists():
            raise FileNotFoundError(path)

        yield from parse_file(path, config)