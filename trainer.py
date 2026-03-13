import numpy as np
from datasets import Value, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# pip install transformers (4.40.2), transformer[torch] ,datasets, scikit-learn, torch, numpy

# 1. LOAD DATA (columns: text, labels, start column is star rating, it's unneccessary, but dataset already has it)

dataset = load_dataset("csv", data_files="data.csv")
dataset = dataset["train"]

# Remove None in text column
dataset = dataset.filter(lambda x: x["text"] is not None)

# 2. CONVERT LABEL STRING -> INT

label_map = {"NEG": 0, "NEU": 1, "POS": 2}  # model can only process numeric labels


def convert_label(example):
    example["labels"] = label_map[example["labels"]]
    return example


dataset = dataset.map(convert_label)

print(type(dataset[0]["labels"]))

# 3. TRAIN / TEST SPLIT

dataset = dataset.train_test_split(test_size=0.2, seed=42)
dataset["test"].to_csv("test_dataset.csv")
# 4. LOAD MODEL

MODEL_NAME = "vinai/phobert-base"  # pre-trained PhoBERT model

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# 5. TOKENIZE


def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )


dataset = dataset.map(tokenize_function, batched=True).cast_column(
    "labels", Value("int64")
)

# 6. REMOVE UNUSED COLUMNS (for Trainer, only input_ids, attention_mask, labels are needed)

dataset = dataset.remove_columns(["text", "start"])

dataset.set_format(type="torch")


# 7. METRICS


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }


# 8. TRAINING ARGUMENTS

training_args = TrainingArguments(
    output_dir="./checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
)

# 9. TRAINER

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)


# 10. TRAIN

trainer.train()

# 11. SAVE MODEL

trainer.save_model("./trained_model")
tokenizer.save_pretrained("./phobert_finetuned")

trainer.evaluate(dataset["test"])