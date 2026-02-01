# IMDb Sentiment Analysis 🎬

## 🚀 Project Overview
This project implements an **end-to-end sentiment analysis pipeline** on the IMDb movie reviews dataset.  
The goal is to predict whether a review is **positive or negative** using both classical ML models and a neural network.

The pipeline includes **dataset download, preprocessing, feature extraction, model training, and saving models & vectorizers for reuse**.

---

## 📁 Dataset
- Dataset: [Stanford IMDb (aclImdb)](https://ai.stanford.edu/~amaas/data/sentiment/)  
- Size: 50,000 reviews (25,000 train + 25,000 test)  
- Labels: `pos` (positive) and `neg` (negative)  
- Automatic download and extraction supported.

---

## 🛠️ Features & Models

**Feature representations:**
- **Bag of Words (BoW)**: unigram + bigram
- **TF-IDF**: unigram + bigram + trigram

**Models:**
- Naive Bayes (MultinomialNB)  
- Neural Network (MLPClassifier)

> Using n-grams allows the models to better capture word sequences and improve prediction accuracy.

---

## 📊 Experimental Results

| Features          | Model | Accuracy | Precision | Recall |
|------------------|-------|----------|-----------|--------|
| BoW (1-2 grams)  | NB    | 0.8073   | 0.8722    | 0.7200 |
| BoW (1-2 grams)  | MLP   | 0.8478   | 0.8605    | 0.8300 |
| TF-IDF (1-3 grams)| NB    | 0.8123   | 0.8715    | 0.7325 |
| TF-IDF (1-3 grams)| MLP   | 0.8345   | 0.8481    | 0.8150 |

> **Analysis:**  
> - MLP achieves higher accuracy and recall.  
> - Naive Bayes achieves slightly higher precision.  
> - Choice of feature representation (BoW vs TF-IDF + n-grams) significantly affects performance.

---

## ⚡ How to Run

### Train Models
```bash
python src/train.py
