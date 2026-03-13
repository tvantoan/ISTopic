import torch
import numpy as np
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "wonrax/phobert-base-vietnamese-sentiment"

# ========================
# LOAD MODEL
# ========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

# ========================
# LOAD TEST DATA
# ========================

dataset = load_dataset("csv", data_files="test_dataset.csv")["train"]

label_map = {"NEG": 0, "NEU": 1, "POS": 2}

def convert_label(example):
    if isinstance(example["labels"], str):
        example["labels"] = label_map[example["labels"]]
    return example

dataset = dataset.map(convert_label)

texts = dataset["text"]
y_true = dataset["labels"]

# ========================
# PREDICT
# ========================

preds = []

for text in tqdm(texts):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()

    preds.append(pred)

# ========================
# METRICS
# ========================

accuracy = accuracy_score(y_true, preds)
precision = precision_score(y_true, preds, average="weighted")
recall = recall_score(y_true, preds, average="weighted")
f1 = f1_score(y_true, preds, average="weighted")

cm = confusion_matrix(y_true, preds)

print("\n===== RESULTS =====")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)

print("\nConfusion Matrix")
print(cm)