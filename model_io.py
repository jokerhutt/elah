import json
import re
import shutil
from dataclasses import asdict
from datetime import datetime
from typing import NamedTuple

import torch
from config import MODEL_DIR, CHECKPOINT_DIR, ModelConfig, TOKENIZER_MODEL
from model.transformer import ElahGPT

from tokenizer.tokenizer import Tokenizer

from safetensors import safe_open
from safetensors.torch import save_model as save_safetensors
from safetensors.torch import load_model as load_safetensors


class Checkpoint(NamedTuple):
    step: int | None
    path: object

    @property
    def legacy(self):
        return self.step is None

    @property
    def label(self):
        size = self.path.stat().st_size / 1e9

        if self.legacy:
            return f"{self.path.name} (legacy, {size:.1f}GB)"

        return f"step {self.step:,} ({self.path.name}, {size:.1f}GB)"


def checkpoint_path(stage, step):
    return CHECKPOINT_DIR / f"{stage}_{step:07d}.pt"


def list_checkpoints(stage):
    if not CHECKPOINT_DIR.exists():
        return []

    pattern = re.compile(rf"^{re.escape(stage)}_(\d+)\.pt$")

    found = [
        Checkpoint(int(match.group(1)), path)
        for path in CHECKPOINT_DIR.iterdir()
        if (match := pattern.match(path.name))
    ]

    found.sort(key=lambda checkpoint: checkpoint.step, reverse=True)

    legacy = CHECKPOINT_DIR / f"{stage}.pt"

    if legacy.exists():
        found.append(Checkpoint(None, legacy))

    return found


def latest_checkpoint(stage):
    found = list_checkpoints(stage)

    return found[0] if found else None


def load_checkpoint(stage, device="cpu", path=None):
    if path is None:
        found = latest_checkpoint(stage)

        if found is None:
            raise FileNotFoundError(f"no checkpoints for stage '{stage}' in {CHECKPOINT_DIR}")

        path = found.path

    checkpoint = torch.load(path, map_location=device)

    model = ElahGPT(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    return model, checkpoint["optimizer"], checkpoint["step"]


def save_checkpoint(model, optimizer, step, stage, config: ModelConfig, keep=None, protect=(), **extra):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    path = checkpoint_path(stage, step)
    tmp = path.with_suffix(".tmp")

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "stage": stage,
            "config": asdict(config),
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            **extra,
        },
        tmp
    )

    tmp.replace(path)

    _prune_checkpoints(stage, keep, {step, *protect})

    return path


def archive_checkpoints(stage):
    found = list_checkpoints(stage)

    if not found:
        return None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = CHECKPOINT_DIR / f"archive_{stage}_{stamp}"
    destination.mkdir(parents=True)

    for checkpoint in found:
        checkpoint.path.rename(destination / checkpoint.path.name)

    return destination


def _prune_checkpoints(stage, keep, protect):
    if keep is None:
        return

    numbered = [c for c in list_checkpoints(stage) if not c.legacy]

    for checkpoint in numbered[keep:]:
        if checkpoint.step not in protect:
            checkpoint.path.unlink(missing_ok=True)

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