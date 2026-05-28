import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.model_selection import GroupShuffleSplit

# ucitavanje podataka
df = pd.read_csv('Projekt1.csv', sep=';', encoding='utf-8')
df['label'] = df['label'].str.lower().str.strip()

#print("Original label values and counts:")
#print(df['label'].value_counts())
#print("\nUnique labels:", df['label'].unique())
#print("Data shape after load:", df.shape)
#print("Columns:", df.columns.tolist())

# Keep only text and label
df = df[['review_id','text', 'label']]

# moramo labele pretvoriti u brojeve
label_mapping = {'positive': 0, 'negative': 1, 'neutral': 2, 'mixed': 3,'sarcasm': 4}

id2label = {v: k for k, v in label_mapping.items()}
label_names_sorted = [id2label[i] for i in sorted(id2label.keys())]

df['label'] = df['label'].map(label_mapping)

# sad micemo sve kaj nije text ili label iz tablice
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# splitanje
# train+val (80%) i test (20%)
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
train_val_idx, test_idx = next(gss1.split(df, groups=df['review_id']))

train_val_df = df.iloc[train_val_idx]
test_df = df.iloc[test_idx]


# iz train+val i validation (20% od train+val)
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
train_idx, val_idx = next(gss2.split(train_val_df,groups=train_val_df['review_id']))

train_df = train_val_df.iloc[train_idx]
val_df = train_val_df.iloc[val_idx]

train_text = train_df['text']
train_labels = train_df['label']

val_text = val_df['text']
val_labels = val_df['label']

test_text = test_df['text']
test_labels = test_df['label']

print(f"Total samples: {len(df)}")
print(f"Training samples: {len(train_text)} ({len(train_text)/len(df)*100:.1f}%)")
print(f"Validation samples: {len(val_text)} ({len(val_text)/len(df)*100:.1f}%)")
print(f"Test samples: {len(test_text)} ({len(test_text)/len(df)*100:.1f}%)")

train_lens = [len(text.split()) for text in train_text]
pad_len = int(np.percentile(train_lens, 95))

print(f"Using max_length = {pad_len}")

# ucitavamo BERT i tokenizer
MODEL_NAME = "classla/bcms-bertic"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModel.from_pretrained(MODEL_NAME)

# Tokenizacija
tokens_train = tokenizer(train_text.tolist(), max_length=pad_len, padding='max_length',truncation=True,return_tensors='pt')
tokens_val = tokenizer(val_text.tolist(),max_length=pad_len, padding='max_length',truncation=True,return_tensors='pt')
tokens_test = tokenizer(test_text.tolist(),max_length=pad_len,padding='max_length',truncation=True,return_tensors='pt')

# Tensori
train_seq = tokens_train['input_ids']
train_mask = tokens_train['attention_mask']
train_y = torch.tensor(train_labels.tolist())

val_seq = tokens_val['input_ids']
val_mask = tokens_val['attention_mask']
val_y = torch.tensor(val_labels.tolist())

test_seq = tokens_test['input_ids']
test_mask = tokens_test['attention_mask']
test_y = torch.tensor(test_labels.tolist())

# DataLoaderi
batch_size = 12

train_data = TensorDataset(train_seq, train_mask, train_y)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_dataloader = DataLoader(val_data, batch_size=batch_size)

test_data = TensorDataset(test_seq, test_mask, test_y)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# zamrzavanje pretrained layera
for name, param in bert.named_parameters():
    if "layer.11" in name or "layer.10" in name or "layer.9" in name or "layer.8" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# BERT klasifikator
class BERT_architecture(nn.Module):

    def __init__(self, bert, num_classes=5):
        super().__init__()

        self.bert = bert
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(
            bert.config.hidden_size,
            num_classes
        )

    def forward(self, sent_id, mask):

        outputs = self.bert(
            sent_id,
            attention_mask=mask
        )

        cls_hs = outputs.last_hidden_state[:, 0]

        x = self.dropout(cls_hs)

        return self.classifier(x)

# model
model = BERT_architecture(bert, num_classes=5)

# optimizer + loss
optimizer = AdamW(model.parameters(), lr=1e-5)

criterion = nn.CrossEntropyLoss()

# trening
def train(model, dataloader, optimizer, criterion, device):

    model.train()

    total_loss = 0
    total_preds = []

    for step, batch in enumerate(dataloader):

        if step % 50 == 0 and step != 0:
            print(f'  Batch {step:>5,} of {len(dataloader):>5,}.')

        sent_id, mask, labels = [r.to(device) for r in batch]
        optimizer.zero_grad()
        preds = model(sent_id, mask)
        loss = criterion(preds, labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_preds.append(preds.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

# evaluacija
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_preds = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step % 50 == 0 and step != 0:
                print(f'  Batch {step:>5,} of {len(dataloader):>5,}.')
            sent_id, mask, labels = [r.to(device) for r in batch]
            preds = model(sent_id, mask)
            loss = criterion(preds, labels)
            total_loss += loss.item()
            total_preds.append(preds.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

# trening modela
epochs = 5

for epoch in range(epochs):

    print(f"Epoch {epoch+1} / {epochs}")

    train_loss, _ = train(model,train_dataloader,optimizer,criterion,device)

    val_loss, _ = evaluate(model, val_dataloader,criterion,device)

    print(f"Training Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")


test_loss, test_preds = evaluate(model,test_dataloader, criterion,device)

test_preds_labels = np.argmax(test_preds, axis=1)

test_true_labels = test_y.numpy()


print("CLASSIFICATION REPORT")

target_names = ['positive','negative','neutral','mixed','sarcasm']

all_labels = [0, 1, 2, 3, 4]

print(classification_report(test_true_labels,test_preds_labels,target_names=target_names,labels=all_labels,zero_division=0))
# Accuracy
accuracy = accuracy_score(test_true_labels,test_preds_labels)
# Weighted Precision
precision = precision_score(test_true_labels,test_preds_labels,average='weighted')
# Weighted Recall
recall = recall_score(test_true_labels,test_preds_labels,average='weighted')
# Weighted F1-score
f1 = f1_score(test_true_labels,test_preds_labels,average='weighted')

print("OVERALL WEIGHTED METRICS")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# Confusion Matrix
label_list = [id2label[i] for i in sorted(id2label.keys())]
cm = confusion_matrix(test_true_labels,test_preds_labels,labels=list(range(len(label_list))))
print("\nConfusion matrix:\n")
print(" " * 12 + " ".join(f"{lbl:12}" for lbl in label_list))
for i, lbl in enumerate(label_list):
    print(f"{lbl:12} " + " ".join(f"{cm[i, j]:12d}" for j in range(len(label_list))))
