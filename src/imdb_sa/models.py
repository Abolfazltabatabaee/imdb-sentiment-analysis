from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier


def build_naive_bayes():
    return MultinomialNB()


def build_mlp(random_state=42):
    return MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=10,
        early_stopping=True,
        n_iter_no_change=2,
        random_state=random_state,
    )
