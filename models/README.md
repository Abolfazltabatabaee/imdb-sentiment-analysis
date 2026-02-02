# Models Directory

This directory is used to store trained models and vectorizers generated during training.

## Contents

- Machine learning models (e.g. Naive Bayes, MLP)
- Feature vectorizers (Bag of Words, TF-IDF)

## Notes

- Trained models and vectorizers are **not committed** to the Git repository.
- All model files (`*.pkl`) are generated automatically by running the training script:
  
  ```bash
  python -m imdb_sa.train
