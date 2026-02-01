from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from .config import MAX_FEATURES

def build_bow_vectorizer():
    
    return CountVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),  
    )

def build_tfidf_vectorizer():
   
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=MAX_FEATURES,
        ngram_range=(1, 3),  
    )
