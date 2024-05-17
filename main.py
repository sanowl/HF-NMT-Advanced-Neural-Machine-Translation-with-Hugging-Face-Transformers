import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset


# Step 1: Data Preparation
dataset_name = "opus100"  # Multilingual dataset from Hugging Face
language_pair = ("en", "fr")  # English to French translation

# Load the dataset
dataset = load_dataset(dataset_name, language_pair)


def preprocess_function(examples):
    return tokenizer(examples["translation"][language_pair[0]], truncation=True, padding=True), tokenizer(examples["translation"][language_pair[1]], truncation=True, padding=True)