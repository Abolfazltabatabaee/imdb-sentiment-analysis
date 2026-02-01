from pathlib import Path

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw"

LIMIT_PER_CLASS = 2000
RANDOM_STATE = 42

MAX_FEATURES = 10000
NGRAM_RANGE = (1, 1)
