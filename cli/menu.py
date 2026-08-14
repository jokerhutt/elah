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
    from dataset import download_datasets, missing_files

    missing = list(missing_files())

    if not missing:
        print("All dataset files are already present")
        return

    names = sorted({name for name, _, _ in missing})
    counts = {name: sum(1 for n, _, _ in missing if n == name) for name in names}

    everything = f"Download all {len(missing)} missing file(s)"
    by_dataset = "Choose datasets"

    how = questionary.select(
        f"{len(missing)} file(s) missing across {len(names)} dataset(s)",
        choices=[everything, by_dataset, "Cancel"],
    ).ask()

    if how is None or how == "Cancel":
        return

    if how == everything:
        download_datasets()
        return

    chosen = questionary.checkbox(
        "Download which datasets?",
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
    from config import STAGES
    from model.trainer import Trainer

    stage = questionary.select("Which stage?", choices=list(STAGES)).ask()

    if stage is None:
        return

    resume_path = _choose_checkpoint(stage)

    if resume_path is _CANCELLED:
        return

    Trainer(stage=stage, resume=resume_path is not None, resume_path=resume_path).run_training()


_CANCELLED = object()


def _choose_checkpoint(stage):
    from model_io import archive_checkpoints, list_checkpoints

    checkpoints = list_checkpoints(stage)

    if not checkpoints:
        return None

    latest = checkpoints[0]

    use_latest = f"Continue from latest - {latest.label}"
    pick = f"Choose from {len(checkpoints)} checkpoint(s)"
    scratch = "Start from scratch"

    how = questionary.select(
        f"'{stage}' has {len(checkpoints)} checkpoint(s)",
        choices=[use_latest, pick, scratch, "Cancel"],
    ).ask()

    if how is None or how == "Cancel":
        return _CANCELLED

    if how == use_latest:
        return latest.path

    if how == pick:
        chosen = questionary.select(
            "Which checkpoint?",
            choices=[questionary.Choice(c.label, value=c.path) for c in checkpoints],
        ).ask()

        return _CANCELLED if chosen is None else chosen

    archive = questionary.confirm(
        f"Archive the {len(checkpoints)} existing checkpoint(s) first?",
        default=True,
    ).ask()

    if archive is None:
        return _CANCELLED

    if archive:
        print(f"Archived to {archive_checkpoints(stage)}")

    return None
