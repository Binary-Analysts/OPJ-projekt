import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -----------------------
# CONFIG
# -----------------------
train_files = ["train grupa 1.csv", "train grupa 2.csv", "train grupa 3.csv", "train grupa 4.csv"]
test_files  = ["test grupa 1.csv", "test grupa 2.csv", "test grupa 3.csv", "test grupa 4.csv"]

VALID_LABELS = ["positive", "negative", "neutral","mixed","sarcasm"]
label_map = {"negative": 0, "neutral": 1, "positive": 2, "mixed":3,"sarcasm":4}


# -----------------------
# HELPERS
# -----------------------
def load_data(file):
    df = pd.read_csv(file, sep=";")
    df = df[["text", "label"]]
    df = df.dropna()
    df["label"] = df["label"].astype(str).str.lower()
    df = df[df["label"].isin(VALID_LABELS)]

    X = df["text"].astype(str).values
    y = np.array([label_map[l] for l in df["label"]])

    return X, y


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


# -----------------------
# RUN
# -----------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

print("\n===== SVM: Individual Training =====")

for i in range(4):
    X_train, y_train = load_data(train_files[i])
    X_test, y_test = load_data(test_files[i])

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    model = LinearSVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\nDataset {i+1}")
    print(evaluate(y_test, y_pred))
