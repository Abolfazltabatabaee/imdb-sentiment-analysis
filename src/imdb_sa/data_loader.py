import tarfile
import urllib.request
from pathlib import Path
import pandas as pd

from .config import IMDB_URL


def ensure_imdb_dataset(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = data_dir / "aclImdb"

    if dataset_dir.exists():
        return dataset_dir

    tgz_path = data_dir / "aclImdb_v1.tar.gz"

    if not tgz_path.exists():
        print(f"Downloading dataset to {tgz_path}")
        urllib.request.urlretrieve(IMDB_URL, tgz_path)

    print("Extracting dataset...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        try:
            tar.extractall(path=data_dir, filter="data")
        except TypeError:
            tar.extractall(path=data_dir)

    return dataset_dir


def load_imdb_split(dataset_dir: Path, split: str, limit_per_class=None) -> pd.DataFrame:
    rows = []

    for folder_name, label_value in [("neg", 0), ("pos", 1)]:
        folder = dataset_dir / split / folder_name
        paths = list(folder.glob("*.txt"))

        if limit_per_class:
            paths = paths[:limit_per_class]

        for p in paths:
            text = p.read_text(encoding="utf-8", errors="ignore")
            rows.append({"text": text, "label": label_value})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df
