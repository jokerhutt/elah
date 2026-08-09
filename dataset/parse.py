import gzip
import json

import pyarrow.parquet as pq


def parse_file(path, config):
    if config.get("format") == "tinystories":
        return _parse_tinystories(path)

    if path.suffixes[-2:] == [".json", ".gz"]:
        return _parse_json_gz(path, config["field"])

    if path.suffix == ".parquet":
        return _parse_parquet(path, config)

    raise ValueError(f"Unsupported file type: {path}")


def _parse_tinystories(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()

    for story in text.split("<|endoftext|>"):
        story = story.strip()

        if story:
            yield story


def _parse_json_gz(path, field):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            value = (json.loads(line).get(field) or "").strip()

            if value:
                yield value


def _parse_parquet(path, config):
    for batch in pq.ParquetFile(path).iter_batches():
        for row in batch.to_pylist():
            sample = _extract_sample(row, config)

            if sample:
                yield sample


def _extract_sample(row, config):
    if config.get("format") == "chat":
        return row["messages"]

    if "fields" in config:
        parts = [row[field].strip() for field in config["fields"] if row.get(field)]

        return "\n\n".join(parts)

    return (row.get(config["field"]) or "").strip()
