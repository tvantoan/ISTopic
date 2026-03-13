import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ===== 1. LOAD MODEL =====

model_path = "./phobert_finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

trainer = Trainer(model=model)

# ===== 2. LOAD TEST DATASET =====

dataset = load_dataset("csv", data_files="test_dataset.csv")
dataset = dataset["train"]

label_map = {"NEG": 0, "NEU": 1, "POS": 2}

def convert_label(example):
    example["labels"] = label_map[example["labels"]]
    return example

# dataset = dataset.map(convert_label)

# ===== 3. TOKENIZE =====

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

dataset = dataset.map(tokenize_function, batched=True)

dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# ===== 4. PREDICT =====

predictions = trainer.predict(dataset)

y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

# ===== 5. METRICS =====

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)

# ===== 6. CONFUSION MATRIX =====

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)