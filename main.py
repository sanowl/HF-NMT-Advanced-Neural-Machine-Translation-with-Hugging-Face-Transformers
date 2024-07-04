from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import json

import torch
import numpy as np
from datasets import concatenate_datasets, load_dataset, DatasetDict, Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, Repository
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    get_scheduler,
)
from sacrebleu.metrics import BLEU
from tqdm import tqdm
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator

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
    use_wandb: bool = True
    wandb_project: str = "multilingual-translation"
    use_fp16: bool = True
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 500
    eval_steps: int = 1000
    save_steps: int = 1000
    max_grad_norm: float = 1.0
    use_8bit_quantization: bool = False
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_accelerate: bool = True
    use_deepspeed: bool = False
    deepspeed_config_path: Optional[str] = "ds_config.json"

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
        datasets = []
        for src_lang, tgt_lang in self.config.language_pairs:
            lang_pair = f"{src_lang}{tgt_lang}"
            try:
                dataset = load_dataset(self.config.dataset_name, lang_pair)['train']
                dataset = dataset.rename_column("translation", f"translation_{src_lang}_{tgt_lang}")
                datasets.append(dataset)
            except Exception as e:
                raise DatasetLoadError(f"Error loading dataset for {lang_pair}: {str(e)}")

        for dataset in datasets:
            dataset = dataset.map(self._align_schema)
        return concatenate_datasets(datasets)

    def _align_schema(self, example: Dict[str, Any]) -> Dict[str, Any]:
        aligned_example = {}
        for src_lang, tgt_lang in self.config.language_pairs:
            aligned_example["translation"] = example.get(f"translation_{src_lang}_{tgt_lang}")
        return aligned_example

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

    def save_tokenizer_to_repo(self) -> None:
        logger.info(f"Saving tokenizer to repository at {self.config.hf_repo_id}...")
        repo = Repository(
            local_dir=str(self.config.output_dir),
            clone_from=f"https://huggingface.co/{self.config.hf_repo_id}",
            use_auth_token=self.config.hf_token
        )
        self._tokenizer.save_pretrained(repo.local_dir)
        repo.git_add(auto_lfs_track=True)
        repo.git_commit("Saved tokenizer")
        repo.git_push()
        logger.info(f"Tokenizer saved to repository at {self.config.hf_repo_id}")

# Model and Trainer class
class TranslationModel:
    def __init__(self, config: TranslationConfig, tokenized_datasets: DatasetDict, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenized_datasets = tokenized_datasets
        self._tokenizer = tokenizer
        self._model: Optional[AutoModelForSeq2SeqLM] = None
        self._trainer: Optional[Seq2SeqTrainer] = None
        self.accelerator: Optional[Accelerator] = None

    def load_model(self) -> None:
        if self.config.use_8bit_quantization:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_checkpoint,
                load_in_8bit=True,
                device_map="auto",
            )
        else:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_checkpoint)

        if self.config.use_peft:
            from peft import get_peft_model, LoraConfig, TaskType

            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
            )
            self._model = get_peft_model(self._model, peft_config)
            self._model.print_trainable_parameters()

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
            fp16=self.config.use_fp16,
            load_best_model_at_end=True,
            metric_for_best_model="eval_bleu",
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            max_grad_norm=self.config.max_grad_norm,
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
            compute_metrics=self.compute_metrics,
        )

        if self.config.use_accelerate:
            self.accelerator = Accelerator()
            self._model, self._trainer.optimizer, self._trainer.lr_scheduler, self._trainer.train_dataloader = self.accelerator.prepare(
                self._model, self._trainer.optimizer, self._trainer.lr_scheduler, self._trainer.train_dataloader
            )

        if self.config.use_deepspeed:
            self._trainer.accelerator.state.deepspeed_plugin.deepspeed_config = json.load(open(self.config.deepspeed_config_path))

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self._tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self._tokenizer.pad_token_id)
        decoded_labels = self._tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu = BLEU()
        result = bleu.corpus_score(decoded_preds, [decoded_labels])

        prediction_lens = [np.count_nonzero(pred != self._tokenizer.pad_token_id) for pred in preds]
        result = {"bleu": result.score, "length": np.mean(prediction_lens)}
        return result

    def train_and_evaluate(self) -> Dict[str, float]:
        if self.config.use_wandb:
            wandb.init(project=self.config.wandb_project, config=self.config.__dict__)
        self._trainer.train()
        return self._trainer.evaluate()

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        inputs = self._tokenizer([f"{src_lang} to {tgt_lang}: {text}"], return_tensors="pt", max_length=self.config.max_input_length, truncation=True, padding="max_length")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        outputs = self._model.generate(**inputs, max_length=self.config.max_target_length, num_beams=4, early_stopping=True)
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save_model(self) -> None:
        save_path = self.config.output_dir / "best_model"
        self._model.save_pretrained(save_path)
        self._tokenizer.save_pretrained(save_path)
        logger.info(f"Model and tokenizer saved to {save_path}")

    def push_to_hub(self) -> None:
        api = HfApi()
        api.upload_folder(
            folder_path=str(self.config.output_dir / "best_model"),
            path_in_repo="",
            repo_id=self.config.hf_repo_id,
            token=self.config.hf_token
        )
        logger.info(f"Model and tokenizer pushed to Hugging Face Hub at {self.config.hf_repo_id}")

        repo = Repository(local_dir=str(self.config.output_dir), clone_from=f"https://huggingface.co/{self.config.hf_repo_id}", use_auth_token=self.config.hf_token)
        repo.git_add(auto_lfs_track=True)
        repo.git_commit("Initial commit")
        repo.git_push()
        logger.info(f"Code pushed to Hugging Face Hub at {self.config.hf_repo_id}")

    def evaluate_on_test_set(self, test_dataset: Dataset) -> Dict[str, float]:
        logger.info("Evaluating model on test set...")
        predictions = self._trainer.predict(test_dataset)
        return self.compute_metrics(predictions)

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
        self.data_processor.load_tokenizer()
        self.data_processor.save_tokenizer_to_repo()  # Save tokenizer to repo immediately after loading

    def run(self) -> None:
        tokenized_datasets = self.data_processor.process_data()
        logger.info("Dataset processing completed.")

        self.translation_model = TranslationModel(self.config, tokenized_datasets, self.data_processor.tokenizer)
        self.translation_model.load_model()
        self.translation_model.setup_trainer()

        logger.info("Starting model training and evaluation...")
        train_results = self.translation_model.train_and_evaluate()
        logger.info(f"Training completed. Results: {train_results}")

        logger.info("Evaluating model on test set...")
        test_results = self.translation_model.evaluate_on_test_set(tokenized_datasets["test"])
        logger.info(f"Test set evaluation results: {test_results}")
