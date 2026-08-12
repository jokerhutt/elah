import questionary


def run():
    actions = {
        "Check dataset files": _check_datasets,
        "Download missing dataset files": _download_datasets,
        "Inspect datasets": _inspect_datasets,
        "Train tokenizer": _train_tokenizer,
        "Pretokenize": _pretokenize,
        "Train": _train,
        "Quit": None,
    }

    while True:
        choice = questionary.select("elah", choices=list(actions)).ask()

        if choice is None or actions[choice] is None:
            return

        try:
            actions[choice]()
        except KeyboardInterrupt:
            print("\nInterrupted")
        except Exception as error:
            print(f"\n{type(error).__name__}: {error}")


def _check_datasets():
    from dataset import check_datasets

    check_datasets()


def _download_datasets():
    from config import DATASETS
    from dataset import download_datasets, missing_files

    missing = list(missing_files())

    if not missing:
        print("All dataset files are already present")
        return

    names = sorted({name for name, _, _ in missing})
    counts = {name: sum(1 for n, _, _ in missing if n == name) for name in names}

    chosen = questionary.checkbox(
        f"{len(missing)} file(s) missing. Download which datasets?",
        choices=[questionary.Choice(f"{name} ({counts[name]})", value=name, checked=True) for name in names],
    ).ask()

    if not chosen:
        return

    download_datasets(chosen)


def _inspect_datasets():
    from dataset import inspect_datasets

    inspect_datasets()


def _train_tokenizer():
    from tokenizer.train import train_tokenizer

    train_tokenizer()


def _pretokenize():
    from tokenizer.pretokenize import pretokenize, pretokenize_pretrain, pretokenize_sft

    which = questionary.select(
        "Pretokenize what?",
        choices=["Both", "Pretrain only", "SFT only"],
    ).ask()

    if which is None:
        return

    {"Both": pretokenize, "Pretrain only": pretokenize_pretrain, "SFT only": pretokenize_sft}[which]()


def _train():
    from config import CHECKPOINT_DIR, STAGES
    from model.trainer import Trainer

    stage = questionary.select("Which stage?", choices=list(STAGES)).ask()

    if stage is None:
        return

    resume = False
    checkpoint = CHECKPOINT_DIR / f"{stage}.pt"

    if checkpoint.exists():
        resume = questionary.confirm(
            f"Continue '{stage}' from {checkpoint.name}?",
            default=True,
        ).ask()

        if resume is None:
            return

    Trainer(stage=stage, resume=resume).run_training()
