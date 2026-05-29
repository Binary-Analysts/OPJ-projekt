import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


train_file = "treniranje_sve_grupe.csv"
test_files  = ["test grupa 1.csv", "test grupa 2.csv", "test grupa 3.csv", "test grupa 4.csv"]

VALID_LABELS = ["positive", "negative", "neutral", "mixed", "sarcasm"]
label_map = {"negative": 0, "neutral": 1, "positive": 2, "mixed":3, "sarcasm":4}


# -----------------------
# LOAD DATA
# -----------------------
def load_data(file):
    df = pd.read_csv(file, sep=";")

    df = df[["text", "label"]].dropna()

    # normalize labels (IMPORTANT)
    df["label"] = df["label"].astype(str).str.lower()

    df = df[df["label"].isin(VALID_LABELS)]

    X = df["text"].astype(str).values
    y = np.array([label_map[l] for l in df["label"]])

    return X, y


# -----------------------
# EVALUATION
# -----------------------
def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


# -----------------------
# TRAIN ON COMBINED SET
# -----------------------
X_train, y_train = load_data(train_file)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train = vectorizer.fit_transform(X_train)

model = LogisticRegression(
    max_iter=1000,
    
)

model.fit(X_train, y_train)

print("\n===== Logistic Regression: Combined Training =====")


# -----------------------
# TEST ON EACH GROUP
# -----------------------
for i, test_file in enumerate(test_files):
    X_test, y_test = load_data(test_file)
    X_test = vectorizer.transform(X_test)

    y_pred = model.predict(X_test)

    print(f"\nTest set {i+1}")
    print(evaluate(y_test, y_pred))
