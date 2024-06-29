"""
Multilingual Translation Model Trainer

This module implements a high-quality, multilingual translation model using the MT5-XL architecture.
It handles data preparation, model training, evaluation, and inference for multiple language pairs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import torch
from datasets import concatenate_datasets, load_dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import HfApi, Repository
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Load environment variables from .env file
load_dotenv()

# Configuration
@dataclass(frozen=True)
class TranslationConfig:
    random_seed: int = 42
    dataset_name: str = "para_crawl"
    language_pairs: Tuple[Tuple[str, str], ...] = (("en", "fr"), ("en", "de"), ("en", "es"))
    model_checkpoint: str = "google/mt5-xl"
    output_dir: Path = field(default_factory=lambda: Path("output"))
    num_proc: int = 4
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_train_epochs: int = 5
    max_input_length: int = 512
    max_target_length: int = 512
    hf_repo_id: str = os.getenv("HF_REPO_ID")
    hf_token: str = os.getenv("HF_TOKEN")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Exception classes
class TranslationError(Exception):
    """Base exception for translation errors."""

class ModelNotInitializedError(TranslationError):
    """Raised when an operation is attempted on an uninitialized model."""

class TokenizerNotInitializedError(TranslationError):
    """Raised when an operation is attempted with an uninitialized tokenizer."""

class DatasetLoadError(TranslationError):
    """Raised when there's an error loading the dataset."""

# Data processing class
class DataProcessor:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self._tokenizer: Optional[AutoTokenizer] = None

    def load_datasets(self) -> DatasetDict:
        datasets = [
            load_dataset(self.config.dataset_name, f"{src_lang}{tgt_lang}")['train']
            for src_lang, tgt_lang in self.config.language_pairs
        ]
        return concatenate_datasets(datasets)

    def preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        inputs = [examples["translation"][src_lang] for src_lang, _ in self.config.language_pairs]
        targets = [examples["translation"][tgt_lang] for _, tgt_lang in self.config.language_pairs]

        model_inputs = self._tokenizer(
            inputs, max_length=self.config.max_input_length, truncation=True, padding="max_length"
        )
        labels = self._tokenizer(
            targets, max_length=self.config.max_target_length, truncation=True, padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def load_tokenizer(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_checkpoint)

    def process_data(self) -> DatasetDict:
        self.load_tokenizer()
        dataset = self.load_datasets()
        tokenized_datasets = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.config.num_proc,
            remove_columns=dataset.column_names
        )
        return tokenized_datasets.train_test_split(test_size=0.1)

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            raise TokenizerNotInitializedError("Tokenizer not initialized. Call 'load_tokenizer' first.")
        return self._tokenizer

# Model and Trainer class
class TranslationModel:
    def __init__(self, config: TranslationConfig, tokenized_datasets: DatasetDict, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenized_datasets = tokenized_datasets
        self._tokenizer = tokenizer
        self._model: Optional[AutoModelForSeq2SeqLM] = None
        self._trainer: Optional[Seq2SeqTrainer] = None

    def load_model(self) -> None:
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_checkpoint)

    def setup_trainer(self) -> None:
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.config.output_dir),
            evaluation_strategy="steps",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=self.config.num_train_epochs,
            predict_with_generate=True,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_bleu",
        )
        data_collator = DataCollatorForSeq2Seq(self._tokenizer, model=self._model)
        self._trainer = Seq2SeqTrainer(
            model=self._model,
            args=training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["test"],
            data_collator=data_collator,
            tokenizer=self._tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

    def train_and_evaluate(self) -> Dict[str, float]:
        self._trainer.train()
        return self._trainer.evaluate()

    def translate(self, text: str, target_lang: str) -> str:
        inputs = self._tokenizer([text], return_tensors="pt", max_length=self.config.max_input_length, truncation=True, padding="max_length")
        outputs = self._model.generate(**inputs, max_length=self.config.max_target_length, num_beams=4, early_stopping=True)
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save_model(self) -> None:
        save_path = self.config.output_dir / "best_model"
        self._model.save_pretrained(save_path)
        self._tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

    def push_to_hub(self) -> None:
        api = HfApi()
        api.upload_folder(
            folder_path=str(self.config.output_dir / "best_model"),
            path_in_repo="",
            repo_id=self.config.hf_repo_id,
            token=self.config.hf_token
        )
        logger.info(f"Model pushed to Hugging Face Hub at {self.config.hf_repo_id}")

        repo = Repository(local_dir=str(self.config.output_dir), clone_from=f"https://huggingface.co/{self.config.hf_repo_id}", use_auth_token=self.config.hf_token)
        repo.git_add(auto_lfs_track=True)
        repo.git_commit("Initial commit")
        repo.git_push()
        logger.info(f"Code pushed to Hugging Face Hub at {self.config.hf_repo_id}")

# Translation pipeline class
class TranslationPipeline:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.data_processor: Optional[DataProcessor] = None
        self.translation_model: Optional[TranslationModel] = None

    def initialize(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(self.config.random_seed)

        self.data_processor = DataProcessor(self.config)
        tokenized_datasets = self.data_processor.process_data()

        self.translation_model = TranslationModel(self.config, tokenized_datasets, self.data_processor.tokenizer)
        self.translation_model.load_model()
        self.translation_model.setup_trainer()

    def run(self) -> None:
        eval_results = self.translation_model.train_and_evaluate()
        logger.info(f"Evaluation results: {eval_results}")

        # Example translation
        english_text = "This is an advanced example sentence to be translated."
        for _, tgt_lang in self.config.language_pairs:
            translated_text = self.translation_model.translate(english_text, tgt_lang)
            logger.info(f"English: {english_text}")
            logger.info(f"{tgt_lang.upper()} Translation: {translated_text}")

        self.translation_model.save_model()
        self.translation_model.push_to_hub()

def main() -> None:
    config = TranslationConfig()
    pipeline = TranslationPipeline(config)

    try:
        pipeline.initialize()
        pipeline.run()
    except TranslationError as e:
        logger.error(f"An error occurred during translation: {str(e)}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
