import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, concatenate_datasets
from sacrebleu import corpus_bleu
from torch.utils.data import DataLoader

# Step 1: Data Preparation
dataset_name = "opus100"
language_pairs = [("en", "fr"), ("en", "de"), ("en", "es")]  # English to French, German, and Spanish

# Load and concatenate the datasets
datasets = []
for language_pair in language_pairs:
    dataset = load_dataset(dataset_name, language_pair)
    datasets.append(dataset)

concatenated_dataset = concatenate_datasets(datasets)

# Preprocess the data
def preprocess_function(examples):
    return tokenizer(examples["translation"][language_pair[0]], truncation=True, padding=True), tokenizer(examples["translation"][language_pair[1]], truncation=True, padding=True)

# Load the tokenizer
model_checkpoint = "t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenize the dataset
tokenized_datasets = concatenated_dataset.map(preprocess_function, batched=True)

# Step 2: Model Selection
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Step 3: Training Setup
batch_size = 32
gradient_accumulation_steps = 4
learning_rate = 1e-4
num_train_epochs = 5

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
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Step 4: Model Training
trainer.train()

# Step 5: Translation and Evaluation
def translate_text(text, target_lang):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    output = model.generate(input_ids, max_length=512, num_beams=8, early_stopping=True, num_return_sequences=4, diversity_penalty=0.5)
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
    test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=data_collator)

    model.eval()
    predictions = []
    references = []

    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=512, num_beams=4)
        predicted_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
        reference_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["labels"]]
        predictions.extend(predicted_texts)
        references.extend(reference_texts)

    bleu_score = corpus_bleu(predictions, [references]).score
    print(f"Test BLEU Score: {bleu_score:.2f}")

evaluate_model()