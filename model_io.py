import json
import shutil
from dataclasses import asdict

import torch
from config import MODEL_DIR, CHECKPOINT_DIR, ModelConfig, TOKENIZER_MODEL
from model.transformer import ElahGPT

from tokenizer.tokenizer import Tokenizer

from safetensors import safe_open
from safetensors.torch import save_model as save_safetensors
from safetensors.torch import load_model as load_safetensors


def load_checkpoint(stage, device="cpu"):
    path = CHECKPOINT_DIR / f"{stage}.pt"
    checkpoint = torch.load(path, map_location=device)

    model = ElahGPT(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    return model, checkpoint["optimizer"], checkpoint["step"]


def save_checkpoint(model, optimizer, step, stage, config: ModelConfig):
    CHECKPOINT_DIR.mkdir(parents = True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{stage}.pt"
    tmp = path.with_suffix(".tmp")

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": asdict(config),
        },
        tmp
    )

    tmp.replace(path)

    return path

def load_model(name, device="cpu"):
    directory = MODEL_DIR / name
    path = directory / "model.safetensors"

    with safe_open(path, framework="pt") as f:
        config = ModelConfig(**json.loads(f.metadata()["config"]))

    model = ElahGPT(config)
    load_safetensors(model, path)

    model.to(device)
    model.eval()

    return model, Tokenizer(path=directory / "tokenizer.model")

def save_model(model, name, config: ModelConfig):
    directory = MODEL_DIR / name
    directory.mkdir(parents=True, exist_ok=True)

    # save model
    save_safetensors(
        model,
        directory / "model.safetensors",
        metadata={"config": json.dumps(asdict(config))}
    )

    # save config for human readable shit
    (directory / "config.json").write_text(json.dumps(asdict(config), indent=2))

    # save tokenizer
    shutil.copyfile(TOKENIZER_MODEL, directory / "tokenizer.model")

    return directory