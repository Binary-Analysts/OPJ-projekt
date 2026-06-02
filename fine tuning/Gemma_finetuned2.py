import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from sklearn.model_selection import StratifiedGroupKFold

#konfiguracija
CSV_PATH = "Projekt1.csv"      
MODEL_NAME = "google/gemma-2-2b"
MAX_LEN = 512
BATCH_SIZE = 4                    
EPOCHS = 3

# ucitavanje dat
df = pd.read_csv(CSV_PATH, sep=';', encoding='utf-8-sig', header=0) 
df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)] 
df = df.dropna(axis=1, how='all')
print("Columns after cleaning:", df.columns.tolist()) 

if 'text' not in df.columns and 'Text' in df.columns:
    df.rename(columns={'Text': 'text'}, inplace=True)       
if 'label' not in df.columns and 'Label' in df.columns:
    df.rename(columns={'Label': 'label'}, inplace=True)

if 'text' not in df.columns or 'label' not in df.columns:
    raise KeyError(f"Missing 'text' or 'label'. Found: {df.columns.tolist()}")   


df = df[['review_id', 'text', 'label']].dropna()                                 
df['label'] = df['label'].astype(str).str.lower().str.strip()  

unique_labels = df['label'].unique()        
print("Unique labels:", unique_labels)

label_list = sorted(unique_labels)   
label2id = {lbl: i for i, lbl in enumerate(label_list)}           
id2label = {i: lbl for lbl, i in label2id.items()}                
df['label_id'] = df['label'].map(label2id)        

#print(f"Label mapping: {label2id}")  
#print(f"Total samples: {len(df)}")

# splitanje
# 20% cijelih rec za testni skup
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

train_val_idx, test_idx = next(
    sgkf.split(df,y=df["label_id"],groups=df["review_id"]))

train_val_df = df.iloc[train_val_idx]
test_df = df.iloc[test_idx]
# drugi skup, tu odvajam opet 20% za validaciju
sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

train_idx, val_idx = next(
    sgkf2.split(train_val_df,y=train_val_df["label_id"],groups=train_val_df["review_id"]))

train_df = train_val_df.iloc[train_idx]
val_df = train_val_df.iloc[val_idx]
# 
#print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}, Test rows: {len(test_df)}")

# 
train_dataset = Dataset.from_pandas(train_df[['text', 'label_id']]).rename_column('label_id', 'labels')  
val_dataset = Dataset.from_pandas(val_df[['text', 'label_id']]).rename_column('label_id', 'labels')
test_dataset = Dataset.from_pandas(test_df[['text', 'label_id']]).rename_column('label_id', 'labels')

# -ucitavamo tokanizer i loru
bnb_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=True)  
tokenizer.pad_token = tokenizer.eos_token  

model = AutoModelForSequenceClassification.from_pretrained( MODEL_NAME,num_labels=len(label_list), id2label=id2label,  label2id=label2id,quantization_config=bnb_config, device_map="auto",   trust_remote_code=True, token=True,)
model.config.pad_token_id = tokenizer.eos_token_id

model = prepare_model_for_kbit_training(model)   
lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, bias="none", task_type="SEQ_CLS") 
model = get_peft_model(model, lora_config) 
model.print_trainable_parameters()

#tokenizacija
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN) 
                                                                                                  
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)                
test_dataset = test_dataset.map(tokenize_function, batched=True)


train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])  
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# mjerenja
def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1) 
    acc = accuracy_score(eval_pred.label_ids, preds) 
    p, r, f1, _ = precision_recall_fscore_support(eval_pred.label_ids, preds, average="weighted") 
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# argumenti za treniranje
training_args = TrainingArguments(
    output_dir="./gemma_finetuned", 
    eval_strategy="epoch", 
    save_strategy="epoch", 
    learning_rate=2e-4, 
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE, 
    num_train_epochs=EPOCHS, 
    weight_decay=0.01,  
    logging_steps=50, 
    load_best_model_at_end=True,    
    metric_for_best_model="accuracy",
    report_to="none",
)
#sad ide trainer
trainer = Trainer(model=model, args=training_args,train_dataset=train_dataset,eval_dataset=val_dataset, compute_metrics=compute_metrics, )

# trening
print("Starting training...")
trainer.train()

#evaluacija
def evaluate_and_confusion(dataset, name):
    print(f"\n{'='*50}\n{name} set\n{'='*50}")
    pred_out = trainer.predict(dataset)
    preds = np.argmax(pred_out.predictions, axis=1)
    labels = pred_out.label_ids

    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    print(f"Accuracy: {acc:.4f}\nPrecision: {p:.4f}\nRecall: {r:.4f}\nF1: {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(labels, preds, labels=list(range(len(label_list))), target_names=label_list,zero_division=0))

    cm = confusion_matrix(labels, preds, labels=list(range(len(label_list))))
    print("\nConfusion matrix:")
    print(" " * 12 + " ".join(f"{lbl:10}" for lbl in label_list))
    for i, lbl in enumerate(label_list):
        print(f"{lbl:12} " + " ".join(f"{cm[i,j]:10d}" for j in range(len(label_list))))
    return cm

val_cm = evaluate_and_confusion(val_dataset, "Validation")
test_cm = evaluate_and_confusion(test_dataset, "Test")

model.save_pretrained("./gemma_finetuned_final")
tokenizer.save_pretrained("./gemma_finetuned_final")
print("Model saved to ./gemma_finetuned_final")
