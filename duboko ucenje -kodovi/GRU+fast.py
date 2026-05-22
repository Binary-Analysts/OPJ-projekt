import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout


FASTTEXT_PATH = "cc.hr.300.vec"


def load_fasttext(path, word_index, dim=300):
    embeddings = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.asarray(parts[1:], dtype="float32")
            embeddings[word] = vec

    vocab_size = len(word_index) + 1
    matrix = np.zeros((vocab_size, dim))

    for word, i in word_index.items():
        vec = embeddings.get(word)
        if vec is not None and i < vocab_size:
            matrix[i] = vec

    return matrix


def build_gru(vocab_size, emb_matrix, num_classes):
    model = Sequential([
        Embedding(vocab_size, 300, weights=[emb_matrix], trainable=False),
        GRU(64),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def run_experiment(train_df, test_df, model_name):

    print("DATASET START")

    train_df["label"] = train_df["label"].astype(str).str.lower().str.strip()
    test_df["label"] = test_df["label"].astype(str).str.lower().str.strip()

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    y_test = le.transform(test_df["label"])

    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["text"])

    X_train = pad_sequences(tokenizer.texts_to_sequences(train_df["text"]), maxlen=100)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_df["text"]), maxlen=100)

    emb_matrix = load_fasttext(FASTTEXT_PATH, tokenizer.word_index)

    model = build_gru(len(tokenizer.word_index) + 1, emb_matrix, len(le.classes_))

    print("TRAINING START")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
    print("TRAINING DONE -> PREDICTION START")

    # spremanje modela
    model.save(f"{model_name}.keras")

    with open(f"{model_name}_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    with open(f"{model_name}_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("EVALUATION START")
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print("EVALUATION DONE")

    return acc, p, r, f1


train_all = pd.read_csv("treniranje_svi.csv", sep=";", engine="python")

results = []

for i, test in enumerate([
    "test grupa 1.csv",
    "test grupa 2.csv",
    "test grupa 3.csv",
    "test grupa 4.csv"
]):
    print(f"\n>>> TRAIN ALL vs TEST {i+1}")

    acc, p, r, f1 = run_experiment(
        train_all,
        pd.read_csv(test, sep=";", engine="python"),
        model_name=f"GRU_ALL_T{i+1}"
    )

    results.append(["GRU", f"ALL-T{i+1}", acc, p, r, f1])


df = pd.DataFrame(results, columns=["Model", "Set", "Accuracy", "Precision", "Recall", "F1"])
print(df)
