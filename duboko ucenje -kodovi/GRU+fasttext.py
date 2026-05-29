import collections
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# 1. Load and Clean Dataset
df = pd.read_csv("Projekt4.csv", sep=";")
df = df[["review_id", "text", "label"]].dropna()
df["label"] = df["label"].str.lower().str.strip()

# Map the 5 Croatian classes
label_mapping = {"negative": 0, "neutral": 1, "positive": 2, "mixed": 3, "sarcasm": 4}
df["label"] = df["label"].map(label_mapping)
df = df.dropna().reset_index(drop=True)

# 2. GroupShuffleSplit (80% Train -> 20% Val, 20% Test)
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
train_idx, test_idx = next(gss_test.split(df, groups=df["review_id"]))
df_train_tmp, df_test = df.iloc[train_idx], df.iloc[test_idx]

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
final_train_idx, val_idx = next(gss_val.split(df_train_tmp, groups=df_train_tmp["review_id"]))
df_train = df_train_tmp.iloc[final_train_idx].reset_index(drop=True)
df_val = df_train_tmp.iloc[val_idx].reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# 3. Simple Tokenization & Vocab Build (Closest to your original style)
def tokenizer(text):
    return str(text).lower().split()

max_length = 256
min_freq = 5
special_tokens = ["<unk>", "<pad>"]

# Build vocab from training tokens
token_counts = collections.Counter([tok for text in df_train["text"] for tok in tokenizer(text)])
vocab_words = [word for word, freq in token_counts.items() if freq >= min_freq]
vocab = special_tokens + vocab_words

word_to_id = {word: idx for idx, word in enumerate(vocab)}
unk_index, pad_index = word_to_id["<unk>"], word_to_id["<pad>"]

# 4. Numericalize Dataset
def process_dataset(dataframe):
    data = []
    for _, row in dataframe.iterrows():
        tokens = tokenizer(row["text"])[:max_length]
        length = max(len(tokens), 1) # Prevent 0 length crashes
        ids = [word_to_id.get(tok, unk_index) for tok in tokens]
        data.append({"ids": torch.tensor(ids, dtype=torch.long), 
                     "length": torch.tensor(length), 
                     "label": torch.tensor(row["label"], dtype=torch.long)})
    return data

train_data = process_dataset(df_train)
valid_data = process_dataset(df_val)
test_data = process_dataset(df_test)

# 5. Collate and Data Loaders (Exactly like your original style)
def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = nn.utils.rnn.pad_sequence([i["ids"] for i in batch], padding_value=pad_index, batch_first=True)
        batch_length = torch.stack([i["length"] for i in batch])
        batch_label = torch.stack([i["label"] for i in batch])
        return {"ids": batch_ids, "length": batch_length, "label": batch_label}
    return collate_fn

batch_size = 256
collate_fn = get_collate_fn(pad_index)

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, collate_fn=collate_fn)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

# 6. GRU Model Architecture (FIXED internal layer names)
class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        # Fixed variable assignment name to self.rnn
        self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout_rate if n_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        embedded = self.dropout(self.embedding(ids))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, length.to("cpu"), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_embedded)
        
        # Fixed structural checks to point to self.rnn
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        else:
            hidden = self.dropout(hidden[-1])
        return self.fc(hidden)

# Initialize Model (5 output classes)
model = GRU(len(vocab), 300, 256, 5, 2, True, 0.5, pad_index)

# 7. Simplified FastText Loading 
try:
    with open("cc.hr.300.vec", "r", encoding="utf-8") as f:
        next(f) # Skip header
        ft_embeddings = {}
        for line in f:
            parts = line.strip().split(" ")
            ft_embeddings[parts[0]] = np.array(parts[1:], dtype=np.float32)
            
    weights = np.random.normal(scale=0.6, size=(len(vocab), 300))
    for idx, word in enumerate(vocab):
        if word in ft_embeddings:
            weights[idx] = ft_embeddings[word]
    weights[pad_index] = np.zeros(300)
    model.embedding.weight.data.copy_(torch.from_numpy(weights))
    print("FastText Embeddings Loaded!")
except FileNotFoundError:
    print("FastText file not found. Training with random weights.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.CrossEntropyLoss().to(device)
model = model.to(device)

# 8. Updated Metrics Engine (Accuracy, Precision, Recall, F1)
def compute_metrics(all_preds, all_labels):
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3, 4])
    
    return acc, prec, rec, f1,cm

# 9. Train and Evaluate Functions
def run_epoch(dataloader, model, criterion, optimizer=None, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
        
    epoch_losses = []
    all_preds, all_labels = [], []
    
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm.tqdm(dataloader, desc="Processing..."):
            ids = batch["ids"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)
            
            prediction = model(ids, length)
            loss = criterion(prediction, label)
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            epoch_losses.append(loss.item())
            all_preds.append(prediction.argmax(dim=-1).cpu().numpy())
            all_labels.append(label.cpu().numpy())
            
    acc, prec, rec, f1,cm = compute_metrics(all_preds, all_labels)
    return np.mean(epoch_losses), acc, prec, rec, f1,cm

# 10. Training Loop
n_epochs = 10
best_valid_loss = float("inf")

for epoch in range(n_epochs):
    train_loss, train_acc, _, _, train_f1, _ = run_epoch(train_data_loader, model, criterion, optimizer, is_train=True)
    valid_loss, valid_acc, _, _, valid_f1, _ = run_epoch(valid_data_loader, model, criterion, is_train=False)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "gru_croatian.pt")
        
    print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.1f}% | Train F1: {train_f1:.2f}")
    print(f"Val Loss: {valid_loss:.3f} | Val Acc: {valid_acc*100:.1f}% | Val F1: {valid_f1:.2f}")

# 11. Final Test Evaluation
model.load_state_dict(torch.load("gru_croatian.pt"))
test_loss, test_acc, test_prec, test_rec, test_f1, test_cm = run_epoch(test_data_loader, model, criterion, is_train=False)

print("\n=== FINAL TEST METRICS ===")
print(f"Accuracy:  {test_acc*100:.2f}%")
print(f"Precision: {test_prec:.3f}")
print(f"Recall:    {test_rec:.3f}")
print(f"F1-Score:  {test_f1:.3f}")

label_list = ["negative", "neutral", "positive", "mixed", "sarcasm"]
print("\nConfusion matrix:")
print(" " * 12 + " ".join(f"{lbl:10}" for lbl in label_list))
for i, lbl in enumerate(label_list):
    print(f"{lbl:12} " + " ".join(f"{test_cm[i, j]:10d}" for j in range(len(label_list))))
    
