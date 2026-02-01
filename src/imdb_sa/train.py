from .config import DATA_DIR, LIMIT_PER_CLASS, RANDOM_STATE
from .data_loader import ensure_imdb_dataset, load_imdb_split
from .features import build_bow_vectorizer, build_tfidf_vectorizer
from .models import build_naive_bayes, build_mlp
from .evaluate import evaluate

import pickle
from pathlib import Path

# مسیر ذخیره مدل‌ها
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def save_model(model, vectorizer, name_prefix):
    """
    مدل و vectorizer را با نام مشخص ذخیره می‌کند.
    """
    model_path = MODEL_DIR / f"{name_prefix}_model.pkl"
    vec_path = MODEL_DIR / f"{name_prefix}_vectorizer.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"✅ Saved {name_prefix} model and vectorizer")


def run_experiment(name, vectorizer, X_train, X_test, y_train, y_test):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results = {}

    # Naive Bayes
    nb = build_naive_bayes()
    nb.fit(X_train_vec, y_train)
    results["NaiveBayes"] = evaluate(y_test, nb.predict(X_test_vec))

    # MLP
    mlp = build_mlp(RANDOM_STATE)
    mlp.fit(X_train_vec, y_train)
    results["MLP"] = evaluate(y_test, mlp.predict(X_test_vec))

    print(f"\n=== Results using {name} ===")
    for model, metrics in results.items():
        print(
            f"{model} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | "
            f"Rec: {metrics['recall']:.4f}"
        )

    # ذخیره مدل‌ها و vectorizer
    save_model(nb, vectorizer, f"{name}_NB")
    save_model(mlp, vectorizer, f"{name}_MLP")

    return results


def main():
    imdb_dir = ensure_imdb_dataset(DATA_DIR)

    train_df = load_imdb_split(imdb_dir, "train", LIMIT_PER_CLASS)
    test_df = load_imdb_split(imdb_dir, "test", LIMIT_PER_CLASS)

    X_train = train_df["text"].values
    y_train = train_df["label"].values
    X_test = test_df["text"].values
    y_test = test_df["label"].values

    # -------- BoW --------
    bow_vectorizer = build_bow_vectorizer()
    run_experiment("Bag of Words", bow_vectorizer, X_train, X_test, y_train, y_test)

    # -------- TF-IDF --------
    tfidf_vectorizer = build_tfidf_vectorizer()
    run_experiment("TF-IDF", tfidf_vectorizer, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
