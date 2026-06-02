import gradio as gr
import torch
import torch.nn as nn
import pickle
import numpy as np
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


# -----------------------------
# BASE PATH
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "zavrsna_verzija")


# -----------------------------
# LOGISTIC REGRESSION
# -----------------------------
with open(os.path.join(DATA_DIR, "lr_demo", "ml_model.pkl"), "rb") as f:
    ml_model = pickle.load(f)

with open(os.path.join(DATA_DIR, "lr_demo", "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

labels = ["negative", "neutral", "positive", "mixed", "sarcasm"]


# -----------------------------
# GRU MODEL
# -----------------------------
class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout_rate, pad_index):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)

        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate if n_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        embedded = self.dropout(self.embedding(ids))

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            length.to("cpu"),
            batch_first=True,
            enforce_sorted=False
        )

        _, hidden = self.rnn(packed)

        if self.rnn.bidirectional:
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=-1)
        else:
            hidden = hidden[-1]

        hidden = self.dropout(hidden)
        return self.fc(hidden)


# load vocab
with open(os.path.join(DATA_DIR, "zavrsni_gru", "vokabular_gru.pkl"), "rb") as f:
    vocab_data = pickle.load(f)

word_to_id = vocab_data["word_to_id"]
max_length = vocab_data["max_length"]
unk_index = vocab_data["unk_index"]
pad_index = vocab_data["pad_index"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


gru_model = GRU(
    vocab_size=len(word_to_id),
    embedding_dim=300,
    hidden_dim=256,
    output_dim=5,
    n_layers=2,
    bidirectional=True,
    dropout_rate=0.5,
    pad_index=pad_index
)

gru_model.load_state_dict(
    torch.load(
        os.path.join(DATA_DIR, "zavrsni_gru", "croatian_gru.pt"),
        map_location=device
    )
)

gru_model.to(device)
gru_model.eval()


# -----------------------------
# GEMMA (PEFT MODEL)
# -----------------------------
gemma_dir = os.path.join(DATA_DIR, "veliki_model")

tokenizer = AutoTokenizer.from_pretrained(gemma_dir)

base_model = AutoModelForSequenceClassification.from_pretrained(
    "google/gemma-2-2b",
    num_labels=5
)

model = PeftModel.from_pretrained(base_model, gemma_dir)

model = model.to(device)
model.eval()


# -----------------------------
# PREDICTIONS
# -----------------------------
def predict_ml(text):
    X = vectorizer.transform([text])
    pred = ml_model.predict(X)[0]
    return labels[pred]


def predict_GRU(text):
    tokens = text.lower().split()
    ids = [word_to_id.get(t, unk_index) for t in tokens][:max_length]

    x = torch.tensor([ids], dtype=torch.long).to(device)
    length = torch.tensor([len(ids)]).to(device)

    with torch.no_grad():
        logits = gru_model(x, length)
        pred = logits.argmax(dim=1).item()

    return labels[pred]


def predict_Gemma(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = logits.argmax(dim=1).item()

    return labels[pred]


def predict_all(text):
    return (
        predict_ml(text),
        predict_GRU(text),
        predict_Gemma(text)
    )


# -----------------------------
# GRADIO UI
# -----------------------------
demo = gr.Interface(
    fn=predict_all,
    inputs=gr.Textbox(label="Upišite neku rečenicu:"),
    outputs=[
        gr.Textbox(label="ML (Logistic Regression)"),
        gr.Textbox(label="GRU (Deep Learning)"),
        gr.Textbox(label="Gemma (Transformer)")
    ],
    title="Analiza sentimenata za hrvatski",
    description="Usporedi odluke za sva tri modela"
)

demo.launch(share=True)
