import urllib.request
from urllib.error import HTTPError, URLError

from tqdm import tqdm

from config import DATASETS


CHUNK_SIZE = 1 << 20


def missing_files(names=None):
    for name in names or DATASETS:
        config = DATASETS[name]

        for filename in config["files"]:
            path = config["root"] / filename

            if not path.exists():
                yield name, filename, path


def download_datasets(names=None):
    missing = list(missing_files(names))

    if not missing:
        print("All dataset files are already present")
        return []

    print(f"{len(missing)} file(s) missing\n")

    failed = []

    for name, filename, path in missing:
        url = f"{DATASETS[name]['base_url']}/{filename}"

        try:
            _download(url, path, f"{name}/{filename}")
        except (HTTPError, URLError, OSError) as error:
            print(f"  FAILED {name}/{filename}: {error}")
            failed.append((name, filename))

    print(f"\nDownloaded {len(missing) - len(failed)}/{len(missing)} file(s)")

    if failed:
        print("Still missing:")
        for name, filename in failed:
            print(f"  {name}/{filename}")

    return failed


def _download(url, path, label):
    path.parent.mkdir(parents=True, exist_ok=True)

    part = path.with_name(path.name + ".part")

    request = urllib.request.Request(url, headers={"User-Agent": "elah"})

    try:
        with urllib.request.urlopen(request) as response:
            total = int(response.headers.get("Content-Length") or 0)

            with open(part, "wb") as f, tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=label,
            ) as progress:
                while chunk := response.read(CHUNK_SIZE):
                    f.write(chunk)
                    progress.update(len(chunk))

        part.replace(path)

    except BaseException:
        part.unlink(missing_ok=True)
        raise
