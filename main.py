import torch
import torch.nn as nn
import logging
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, get_scheduler, EarlyStoppingCallback
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

# Step 1: Data Preparation
dataset_name = "para_crawl"
language_pairs = [("en", "fr"), ("en", "de"), ("en", "es")]  # English to French, German, and Spanish

# Load and filter the datasets
datasets = []
for language_pair in language_pairs:
    src_lang, tgt_lang = language_pair
    lang_pair_config = f"{src_lang}{tgt_lang}"
    dataset = load_dataset(dataset_name, lang_pair_config)
    datasets.append(dataset)

# Concatenate and filter the datasets for the desired language pairs
filtered_datasets = [dataset.filter(lambda x: x["lang_pair"] == f"{src_lang}-{tgt_lang}" or x["lang_pair"] == f"{tgt_lang}-{src_lang}")
                     for dataset, (src_lang, tgt_lang) in zip(datasets, language_pairs)]
concatenated_dataset = concatenate_datasets(filtered_datasets)
# Preprocess the data
def preprocess_function(examples):
    inputs = [examples["translation"][language_pair[0]] for language_pair in language_pairs]
    targets = [examples["translation"][language_pair[1]] for language_pair in language_pairs]
    model_inputs = tokenizer(inputs, truncation=True, padding=True)
    labels = tokenizer(targets, truncation=True, padding=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Load the tokenizer
model_checkpoint = "google/mt5-xl"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenize the dataset
tokenized_datasets = concatenated_dataset.map(preprocess_function, batched=True, num_proc=4, remove_columns=concatenated_dataset.column_names)

# Step 2: Model Selection
class CustomMT5(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_ids, attention_mask, labels):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        decoder_outputs = self.decoder(input_ids=labels, encoder_outputs=encoder_outputs.last_hidden_state)
        return decoder_outputs.logits

# Determine the device to use based on availability
if torch.cuda.is_available() and torch.backends.mps.is_available():
    device = torch.device("mps")  # Use M1 if available
elif torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA if available
else:
    device = torch.device("cpu")  # Use CPU if no GPU or M1 is available

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
custom_model = CustomMT5(model.encoder, model.decoder)
custom_model.to(device)  # Move the model to the device

# Step 3: Training Setup
def objective(trial):
    batch_size = trial.suggest_int('batch_size', 8, 64)
    gradient_accumulation_steps = trial.suggest_int('gradient_accumulation_steps', 1, 8)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
    num_train_epochs = trial.suggest_int('num_train_epochs', 3, 10)
    warmup_steps = trial.suggest_int('warmup_steps', 500, 2000)

    args = Seq2SeqTrainingArguments(
        output_dir="output",
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
        fp16=True,  # Enable mixed precision training
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="eval_bleu",  # Use BLEU score for best model selection
        evaluation_metrics=["bleu", "rouge"],  # Evaluate on BLEU and ROUGE scores
        early_stopping_patience=3,  # Stop training if no improvement in 3 epochs
        early_stopping_threshold=0.001,  # Minimum improvement required
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=custom_model)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(custom_model.parameters(), lr=learning_rate)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.max_steps)

    # Trainer
    trainer = Seq2SeqTrainer(
        model=custom_model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        optimizers=(optimizer, scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=args.early_stopping_threshold)],
    )

    # Step 4: Model Training
    trainer.train()

    # Evaluate on validation dataset
    eval_results = trainer.evaluate()
    validation_bleu = eval_results['eval_bleu']
    return validation_bleu

# Hyperparameter tuning with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Print the best hyperparameters
print('Best Hyperparameters:')
print(f'Batch Size: {study.best_params["batch_size"]}')
print(f'Gradient Accumulation Steps: {study.best_params["gradient_accumulation_steps"]}')
print(f'Learning Rate: {study.best_params["learning_rate"]}')
print(f'Number of Training Epochs: {study.best_params["num_train_epochs"]}')
print(f'Warmup Steps: {study.best_params["warmup_steps"]}')

# Step 5: Translation and Evaluation
def translate_text(text, target_lang):
    input_ids = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)["input_ids"].to(device)
    output = custom_model.generate(input_ids, max_length=512, num_beams=8, early_stopping=True, num_return_sequences=4, diversity_penalty=0.5)
    translated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
    best_translation = max(translated_texts, key=lambda x: corpus_bleu([x], [[text]]).score)
    return best_translation

# Example usage
english_text = "This is an advanced example sentence to be translated."
target_languages = ["fr", "de", "es"]
for lang in target_languages:
    translated_text = translate_text(english_text, lang)
    print(f"English: {english_text}")
    print(f"{lang.upper()} Translation: {translated_text}\n")

# Evaluate on test dataset
def evaluate_model():
    test_dataset = tokenized_datasets["test"]
    test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=data_collator, num_workers=4)
    custom_model.eval()
    predictions = []
    references = []
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

evaluate_model()

# Save the model checkpoint
save_path = os.path.join("output", "best_model")
custom_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)