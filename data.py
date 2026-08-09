import gzip
import json
from pathlib import Path

import pyarrow.parquet as pq

from config import DATASETS

def check_datasets():
    for name, config in DATASETS.items():
        root = config["root"]

        print(f"\n{name}")
        print(f"Directory: {'exists' if root.exists() else 'does not exist'}")

        for filename in config["files"]:
            path = root / filename
            print(f"  {'✓' if path.exists() else '✗'} {filename}")


def inspect_file(path: Path):
    if path.suffix == ".parquet":
        table = pq.read_table(path)
        print("Columns:", table.column_names)
        print("First row:", table.slice(0, 1).to_pylist()[0])

    elif path.suffixes[-2:] == [".json", ".gz"]:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            row = json.loads(next(f))

        print("Keys:", list(row.keys()))
        print("First row:", row)

    elif path.suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read(1000)

        print("First 1000 chars:")
        print(text)

    else:
        print(f"Unsupported file type: {path}")


def inspect_datasets():
    for name, config in DATASETS.items():
        path = config["root"] / config["files"][0]

        print(f"\n{'=' * 60}")
        print(name)
        print(path)
        print("=" * 60)

        inspect_file(path)