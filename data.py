

from config import DATASETS

for name, config in DATASETS.items():
    for filename in config["files"]:
        path = config["root"] / filename

        if not path.exists():
            print(f"Missing: {name}/{filename}")