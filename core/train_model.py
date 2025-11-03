import os
import pandas as pd
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import transformers
import accelerate
import torch

# ===== 0️⃣ Version Check =====
print(f"Transformers: {transformers.__version__}, Accelerate: {accelerate.__version__}, Torch: {torch.__version__}")

# ===== 1️⃣ Setup =====
MODEL_NAME = "google/flan-t5-small"
DATA_PATH = "synthetic_dataset.csv"
OUTPUT_DIR = "./Sales_pitch/sales_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 2️⃣ Load Dataset =====
print("📦 Loading dataset...")
df = pd.read_csv(DATA_PATH).fillna("")
dataset = Dataset.from_pandas(df)
print(f"✅ Loaded {len(df)} samples.")

# ===== 3️⃣ Tokenizer =====
print("🔤 Loading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

def preprocess(example):
    input_text = (
        f"Product: {example['product_name']}\n"
        f"Description: {example['description']}\n"
        f"Features: {example['features']}\n"
        f"Generate sales recommendation:"
    )
    output_text = (
        f"Target audience: {example['target_audience']}. "
        f"Highlight: {example['highlight_features']}. "
        f"Sales strategy: {example['sales_strategy']}."
    )
    model_inputs = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    labels = tokenizer(
        output_text,
        truncation=True,
        padding="max_length",
        max_length=64
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("🧩 Tokenizing dataset...")
tokenized = dataset.map(preprocess, batched=False)
print("✅ Tokenization complete.")

# ===== 4️⃣ Model =====
print("⚙️ Loading model...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Freeze most layers for lightweight training
for param in model.parameters():
    param.requires_grad = False
for param in model.lm_head.parameters():
    param.requires_grad = True

# ===== 5️⃣ Training Args =====
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    save_strategy="no",
    logging_steps=5,
    learning_rate=5e-4,
    disable_tqdm=True,
    report_to="none",
    no_cuda=True,  # force CPU mode, avoids accelerate unwrap conflicts
)

# ===== 6️⃣ Trainer =====
print("🚀 Training started (should take ~2 minutes on CPU)...")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
)

# 🚑 Clean Accelerate Patch
if hasattr(trainer, "accelerator") and trainer.accelerator is not None:
    orig_unwrap = trainer.accelerator.unwrap_model
    # redefine unwrap_model without unsupported args
    def safe_unwrap_model(model, *_, **__):
        return model
    trainer.accelerator.unwrap_model = safe_unwrap_model

trainer.train()

# ===== 7️⃣ Save Model =====
print("💾 Saving fine-tuned model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Training complete. Model saved to {OUTPUT_DIR}")
