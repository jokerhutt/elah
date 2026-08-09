from pathlib import Path

# Data Paths
DATA_ROOT = Path.home() / "data" / "training"
PRETRAIN_ROOT = DATA_ROOT
SFT_ROOT = DATA_ROOT / "sft"

DATASETS = {
    "fineweb_edu": {
        "root": DATA_ROOT / "fineweb-edu",
        "files": [
            f"{i:03d}_00000.parquet"
            for i in range(10)
        ],
    },

    "tinystories": {
        "root": DATA_ROOT / "tinystories",
        "files": [
            "TinyStories-train.txt",
            "TinyStories-valid.txt",
        ],
    },

    "codeparrot": {
        "root": DATA_ROOT / "codeparrot",
        "files": [
            f"file-{i:012d}.json.gz"
            for i in range(1, 49)
            if i not in {21, 29, 45}
        ],
    },

    "tulu3": {
        "root": SFT_ROOT / "tulu3",
        "files": [
            "train-00000-of-00006.parquet",
            "train-00001-of-00006.parquet",
        ],
    },
}