import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "phobert_finetuned")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()

score_map = {0: 0, 1: 2.5, 2: 5} 


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)[0]  
    score = 0
    for i, p in enumerate(probs):
        score += score_map[i] * p.item()

    return round(score, 2)
