import gzip
import json

import pyarrow.parquet as pq

from config import DATASETS


def inspect_datasets():
    for name, config in DATASETS.items():
        path = config["root"] / config["files"][0]

        print(f"\n{'=' * 60}")
        print(name)
        print(path)
        print("=" * 60)

        _inspect_file(path)


def _inspect_file(path):
    if path.suffix == ".parquet":
        table = pq.read_table(path)

        print("Columns:", table.column_names)
        print("First row:", table.slice(0, 1).to_pylist()[0])
        return

    if path.suffixes[-2:] == [".json", ".gz"]:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            row = json.loads(next(f))

        print("Keys:", list(row.keys()))
        print("First row:", row)
        return

    if path.suffix == ".txt":
        with open(path, encoding="utf-8") as f:
            print("First 1000 chars:")
            print(f.read(1000))
        return

    print(f"Unsupported file type: {path}")
