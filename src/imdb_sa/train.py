from __future__ import annotations

import pickle
from pathlib import Path

from .config import BASE_DIR, DATA_DIR, LIMIT_PER_CLASS, RANDOM_STATE
from .data_loader import ensure_imdb_dataset, load_imdb_split
from .evaluate import evaluate
from .features import build_bow_vectorizer, build_tfidf_vectorizer
from .models import build_mlp, build_naive_bayes

# مسیر ذخیره مدل‌ها (ریشه پروژه /models)
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_model(model, vectorizer, name_prefix: str) -> None:
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


def _print_results_table(experiment_name: str, results: dict) -> None:
    """
    چاپ نتایج به صورت جدول مرتب
    """
    print(f"\n============= Results using {experiment_name} ============= ")

    headers = ["Model", "Acc", "Prec", "Rec"]
    rows = []
    for model_name, metrics in results.items():
        rows.append(
            [
                model_name,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
            ]
        )

    # محاسبه عرض ستون‌ها
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells):
        return " | ".join(cells[i].ljust(col_widths[i]) for i in range(len(cells)))

    sep = "-+-".join("-" * w for w in col_widths)

    print(fmt_row(headers))
    print(sep)
    for r in rows:
        print(fmt_row(r))


def _print_confusion_matrices(results: dict) -> None:
    for model_name, metrics in results.items():
        tn, fp, fn, tp = metrics["confusion_matrix"].ravel()

        print(f"\nConfusion Matrix — {model_name}\n")
        print("            Pred -    Pred +")
        print("          ┌────────┬────────┐")
        print(f"Actual  - │ {tn:^6} │ {fp:^6} │")
        print("          ├────────┼────────┤")
        print(f"Actual  + │ {fn:^6} │ {tp:^6} │")
        print("          └────────┴────────┘")


def run_experiment(name: str, vectorizer, X_train, X_test, y_train, y_test) -> dict:
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

    # چاپ نتایج به شکل جدول
    _print_results_table(name, results)

    # چاپ Confusion Matrix ها
    _print_confusion_matrices(results)

    # ذخیره مدل‌ها و vectorizer (نام امن بدون فاصله)
    safe_name = name.replace(" ", "_")
    save_model(nb, vectorizer, f"{safe_name}_NB")
    save_model(mlp, vectorizer, f"{safe_name}_MLP")

    return results


def main() -> None:
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
