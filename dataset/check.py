from config import DATASETS


def check_datasets():
    for name, config in DATASETS.items():
        _check_dataset(name, config)


def _check_dataset(name, config):
    root = config["root"]
    files = config["files"]

    print(f"\n{name}")
    print(f"Directory: {'exists' if root.exists() else 'does not exist'}")

    present = 0

    for filename in files:
        exists = (root / filename).exists()

        print(f"  {'✓' if exists else '✗'} {filename}")

        if exists:
            present += 1

    print(f"\n{present}/{len(files)} files present")
