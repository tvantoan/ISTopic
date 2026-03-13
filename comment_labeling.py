import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "./phobert_finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()

label_map = {0: "negative", 1: "neutral", 2: "positive"}

while True:
    text = input("Enter text (or 'exit' to quit): ")
    if text.lower() == "exit":
        break

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    probs = F.softmax(logits, dim=1)[0]

    pred = torch.argmax(probs).item()

    print("\nPrediction:", label_map[pred])
    # print("Score:", probs[pred].item())

    print("\nProbabilities:")
    for i in range(3):
        print(f"{label_map[i]}: {probs[i].item()*100:.2f}%")

    print("-" * 40)
