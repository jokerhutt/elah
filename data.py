

from config import DATASETS

def check_datasets():
    for name, config in DATASETS.items():
        root = config["root"]

        print(f"\n{name}")
        print(f"Directory: {'exists' if root.exists() else 'does not exist'}")

        for filename in config["files"]:
            path = root / filename
            print(f"  {'✓' if path.exists() else '✗'} {filename}")