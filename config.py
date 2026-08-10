from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
PRETRAIN_ROOT = DATA_ROOT
SFT_ROOT = DATA_ROOT / "sft"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

# DATA LOADING
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
END_OF_TEXT = "<|endoftext|>"
SPECIAL_TOKENS = [IM_START, IM_END, END_OF_TEXT]

# TOKENIZER CONFIG
VOCAB_SIZE = 32_000
MODEL_TYPE = "bpe"

# EMBEDDING CONFIG
D_MODEL = 1024

TOKENIZER_DIR = ARTIFACTS_ROOT / "tokenizer"
TOKENIZER_PREFIX = TOKENIZER_DIR / "tokenizer"
TOKENIZER_MODEL = TOKENIZER_PREFIX.with_suffix(".model")
TOKENIZER_VOCAB = TOKENIZER_PREFIX.with_suffix(".vocab")

# TRANSFORMER CONFIG
BLOCK_SIZE = 255
BATCH_SIZE = 128
MAX_ITERS = 5000

LEARNING_RATE = 3e-4
NORM_EPS = 1e-5
N_HEAD = 8
N_LAYER = 8
DROPOUT = 0.2

TRAINING_SPLIT_PERCENTAGE = 0.9

# LOGGING
SAMPLE_INTERVAL = 2000
CHECKPOINT_INTERVAL = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 40000

# GPU
GPU_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# PRETOKENIZED DATA
TOKENS_DIR = ARTIFACTS_ROOT / "tokens"
PRETRAIN_TOKENS = TOKENS_DIR / "pretrain.bin"
SFT_TOKENS = TOKENS_DIR / "sft.bin"
TOKEN_DTYPE = "uint16"

# DATASETS
DATASETS = {
    "fineweb_edu": {
        "root": PRETRAIN_ROOT / "fineweb-edu",
        "base_url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT",
        "field": "text",
        "files": [
            f"{i:03d}_00000.parquet"
            for i in range(10)
        ],
    },

    "tinystories": {
        "root": PRETRAIN_ROOT / "tinystories",
        "base_url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main",
        "format": "tinystories",
        "files": [
            "TinyStories-train.txt",
        ],
    },

    "codeparrot": {
        "root": PRETRAIN_ROOT / "codeparrot",
        "base_url": "https://huggingface.co/datasets/codeparrot/codeparrot-clean/resolve/main",
        "field": "content",
        "files": [
            f"file-{i:012d}.json.gz"
            for i in range(1, 49)
            if i not in {21, 29, 45}
        ],
    },

    "cosmopedia_auto_math": {
        "root": PRETRAIN_ROOT / "cosmopedia" / "data" / "auto_math_text",
        "base_url": "https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/resolve/main/data/auto_math_text",
        "fields": ["prompt", "text"],
        "files": [
            f"train-{i:05d}-of-00018.parquet"
            for i in range(5)
        ],
    },

    "cosmopedia_khanacademy": {
        "root": PRETRAIN_ROOT / "cosmopedia" / "data" / "khanacademy",
        "base_url": "https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/resolve/main/data/khanacademy",
        "fields": ["prompt", "text"],
        "files": [
            "train-00000-of-00001.parquet",
        ],
    },

    "cosmopedia_openstax": {
        "root": PRETRAIN_ROOT / "cosmopedia" / "data" / "openstax",
        "base_url": "https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/resolve/main/data/openstax",
        "fields": ["prompt", "text"],
        "files": [
            "train-00000-of-00002.parquet",
            "train-00001-of-00002.parquet",
        ],
    },

    "cosmopedia_stanford": {
        "root": PRETRAIN_ROOT / "cosmopedia" / "data" / "stanford",
        "base_url": "https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/resolve/main/data/stanford",
        "fields": ["prompt", "text"],
        "files": [
            f"train-{i:05d}-of-00013.parquet"
            for i in range(10)
        ],
    },

    "cosmopedia_stories": {
        "root": PRETRAIN_ROOT / "cosmopedia" / "data" / "stories",
        "base_url": "https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/resolve/main/data/stories",
        "fields": ["prompt", "text"],
        "files": [
            f"train-{i:05d}-of-00043.parquet"
            for i in range(10)
        ],
    },

    "cosmopedia_wikihow": {
        "root": PRETRAIN_ROOT / "cosmopedia" / "data" / "wikihow",
        "base_url": "https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/resolve/main/data/wikihow",
        "fields": ["prompt", "text"],
        "files": [
            "train-00000-of-00002.parquet",
            "train-00001-of-00002.parquet",
        ],
    },

    "open_web_math": {
        "root": PRETRAIN_ROOT / "open-web-math",
        "base_url": "https://huggingface.co/datasets/open-web-math/open-web-math/resolve/main/data",
        "field": "text",
        "files": [
            "train-00000-of-00114-5a023365406cb9c4.parquet",
            "train-00001-of-00114-e32fc2813a15f61c.parquet",
            "train-00002-of-00114-1429d96b99aec578.parquet",
            "train-00003-of-00114-e7fc257ef044bc03.parquet",
            "train-00004-of-00114-3158c787ea8296d3.parquet",
            "train-00005-of-00114-c525c7efee442287.parquet",
            "train-00006-of-00114-c82ec070af45d226.parquet",
            "train-00007-of-00114-36c74b525c9694d4.parquet",
            "train-00008-of-00114-bf41cf8843148a70.parquet",
            "train-00009-of-00114-691ac94b115fea46.parquet",
            "train-00010-of-00114-5805e25b4884966e.parquet",
            "train-00011-of-00114-da8ee2fcf07be148.parquet",
            "train-00012-of-00114-7252b11ca4b39acd.parquet",
            "train-00013-of-00114-a189dcaf5ac68c7e.parquet",
            "train-00014-of-00114-23a118ee3aaea5c3.parquet",
            "train-00015-of-00114-e65817847eac684c.parquet",
            "train-00016-of-00114-7b0ca70e75bb60ee.parquet",
            "train-00017-of-00114-7680a1785b342d09.parquet",
            "train-00018-of-00114-f187dd9c797b315c.parquet",
            "train-00019-of-00114-95e7ebe4402c9bfb.parquet",
            "train-00020-of-00114-49f2b2f31d348847.parquet",
            "train-00021-of-00114-64c103f9fbdf2cf4.parquet",
            "train-00022-of-00114-4d18242ef5fd3198.parquet",
            "train-00023-of-00114-9ec2a6a02bf1d9d0.parquet",
        ],
    },

    "tulu3": {
        "root": SFT_ROOT / "tulu3",
        "base_url": "https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/resolve/main/data",
        "format": "chat",
        "files": [
            "train-00000-of-00006.parquet",
            "train-00001-of-00006.parquet",
        ],
    },
}