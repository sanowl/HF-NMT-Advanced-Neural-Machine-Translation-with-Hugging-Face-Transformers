import torch
import torch.nn as nn
import logging
import random
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    get_scheduler,
    EarlyStoppingCallback
)
from datasets import load_dataset, concatenate_datasets
from sacrebleu import corpus_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Constants
DATASET_NAME = "para_crawl"
LANGUAGE_PAIRS = [("en", "fr"), ("en", "de"), ("en", "es")]
MODEL_CHECKPOINT = "google/mt5-xl"
OUTPUT_DIR = "output"
NUM_PROC = 4
BATCH_SIZE = 32

# Step 1: Data Preparation
def load_and_concatenate_datasets(dataset_name, language_pairs):
    datasets = []
    for src_lang, tgt_lang in language_pairs:
        dataset = load_dataset(dataset_name, f"{src_lang}{tgt_lang}")
        filtered_dataset = dataset.filter(lambda x: x["lang_pair"] in [f"{src_lang}-{tgt_lang}", f"{tgt_lang}-{src_lang}"])
        datasets.append(filtered_dataset)
    return concatenate_datasets(datasets)

def preprocess_function(examples, tokenizer, language_pairs):
    inputs = [examples["translation"][src_lang] for src_lang, _ in language_pairs]
    targets = [examples["translation"][tgt_lang] for _, tgt_lang in language_pairs]
    model_inputs = tokenizer(inputs, truncation=True, padding=True)
    labels = tokenizer(targets, truncation=True, padding=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Load and concatenate datasets
concatenated_dataset = load_and_concatenate_datasets(DATASET_NAME, LANGUAGE_PAIRS)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# Tokenize the dataset
tokenized_datasets = concatenated_dataset.map(
    lambda examples: preprocess_function(examples, tokenizer, LANGUAGE_PAIRS),
    batched=True,
    num_proc=NUM_PROC,
    remove_columns=concatenated_dataset.column_names
)

# Step 2: Model Selection
class CustomMT5(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.decoder = model.decoder

    def forward(self, input_ids, attention_mask, labels):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        decoder_outputs = self.decoder(input_ids=labels, encoder_outputs=encoder_outputs.last_hidden_state)
        return decoder_outputs.logits

def get_device():
    if torch.cuda.is_available() and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Load model and move to device
device = get_device()
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
custom_model = CustomMT5(base_model).to(device)

# Step 3: Training Setup
def objective(trial):
    # Hyperparameter tuning
    batch_size = trial.suggest_int('batch_size', 8, 64)
    gradient_accumulation_steps = trial.suggest_int('gradient_accumulation_steps', 1, 8)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
    num_train_epochs = trial.suggest_int('num_train_epochs', 3, 10)
    warmup_steps = trial.suggest_int('warmup_steps', 500, 2000)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=0.01,
        save_total_limit=5,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        warmup_steps=warmup_steps,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_bleu",
        evaluation_metrics=["bleu", "rouge"],
        early_stopping_patience=3,
        early_stopping_threshold=0.001,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=custom_model)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(custom_model.parameters(), lr=learning_rate)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_args.max_steps)

    # Trainer
    trainer = Seq2SeqTrainer(
        model=custom_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        optimizers=(optimizer, scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience, early_stopping_threshold=training_args.early_stopping_threshold)],
    )

    # Train model
    trainer.train()

    # Evaluate model
    eval_results = trainer.evaluate()
    return eval_results['eval_bleu']

# Hyperparameter tuning with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Print best hyperparameters
print('Best Hyperparameters:')
for key, value in study.best_params.items():
    print(f'{key}: {value}')

# Step 5: Translation and Evaluation
def translate_text(text, target_lang):
    input_ids = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)["input_ids"].to(device)
    output = custom_model.generate(input_ids, max_length=512, num_beams=8, early_stopping=True, num_return_sequences=4, diversity_penalty=0.5)
    translated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
    best_translation = max(translated_texts, key=lambda x: corpus_bleu([x], [[text]]).score)
    return best_translation

# Example usage
def example_translation():
    english_text = "This is an advanced example sentence to be translated."
    target_languages = ["fr", "de", "es"]
    for lang in target_languages:
        translated_text = translate_text(english_text, lang)
        print(f"English: {english_text}")
        print(f"{lang.upper()} Translation: {translated_text}\n")

# Evaluate on test dataset
def evaluate_model():
    test_dataset = tokenized_datasets["test"]
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=DataCollatorForSeq2Seq(tokenizer, model=custom_model), num_workers=NUM_PROC)
    custom_model.eval()
    predictions, references = [], []
    progress_bar = tqdm(total=len(test_dataloader), desc="Evaluating", unit="batch")

    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            outputs = custom_model.generate(input_ids, attention_mask=attention_mask, max_length=512, num_beams=4)
        predicted_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
        reference_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]
        predictions.extend(predicted_texts)
        references.extend(reference_texts)
        progress_bar.update(1)

    progress_bar.close()
    bleu_score = corpus_bleu(predictions, [references]).score
    rouge_score = corpus_bleu(predictions, [references], rouge_scorer=True).score
    print(f"Test BLEU Score: {bleu_score:.2f}")
    print(f"Test ROUGE Score: {rouge_score:.2f}")

# Save the model checkpoint
def save_model():
    save_path = os.path.join(OUTPUT_DIR, "best_model")
    custom_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    example_translation()
    evaluate_model()
    save_model()
