import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import Repository
from sacrebleu.metrics import BLEU
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

# Additional imports
from rouge_score import rouge_scorer
from langdetect import detect

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONFIG_PATH = "config.json"

# Custom exceptions
class TranslationError(Exception):
    pass

class ModelNotInitializedError(TranslationError):
    pass

class TokenizerNotInitializedError(TranslationError):
    pass

class DatasetLoadError(TranslationError):
    pass

# Function to get dynamic batch size
def get_dynamic_batch_size() -> int:
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        if total_memory > 20e9:
            return 32
        elif total_memory > 10e9:
            return 16
        else:
            return 8
    return 4  # Default for CPU

# Configuration dataclass
@dataclass
class TranslationConfig:
    random_seed: int = 42
    dataset_name: str = "para_crawl"
    language_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [("en", "fr"), ("en", "de"), ("en", "es")])
    model_checkpoint: str = "google/mt5-small"  # Changed to smaller model for testing
    output_dir: Path = field(default_factory=lambda: Path("output"))
    num_proc: int = 4
    batch_size: int = field(default_factory=get_dynamic_batch_size)
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    max_input_length: int = 128
    max_target_length: int = 128
    hf_repo_id: Optional[str] = None
    hf_token: Optional[str] = None
    use_wandb: bool = False
    wandb_project: str = "multilingual-translation"
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 500
    max_grad_norm: float = 1.0
    use_8bit_quantization: bool = False
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_accelerate: bool = True
    mixed_precision: str = "fp16"
    early_stopping_patience: int = 3
    scheduler_type: str = "linear"
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    clipping_threshold: float = 1.0

    def __post_init__(self):
        # Load environment variables if not set
        if not self.hf_repo_id:
            self.hf_repo_id = os.getenv("HF_REPO_ID")
        if not self.hf_token:
            self.hf_token = os.getenv("HF_TOKEN")

# Data Processor class
class DataProcessor:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.tokenizer: Optional[AutoTokenizer] = None

    def load_tokenizer(self) -> None:
        logger.info("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_checkpoint)
            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise TokenizerNotInitializedError(f"Error loading tokenizer: {str(e)}")

    def save_tokenizer_to_repo(self) -> None:
        if not self.config.hf_repo_id or not self.config.hf_token:
            logger.warning("HF_REPO_ID or HF_TOKEN not provided. Skipping tokenizer upload.")
            return
        logger.info(f"Saving tokenizer to repository at {self.config.hf_repo_id}...")
        try:
            repo = Repository(
                local_dir=str(self.config.output_dir),
                clone_from=f"https://huggingface.co/{self.config.hf_repo_id}",
                use_auth_token=self.config.hf_token
            )
            self.tokenizer.save_pretrained(repo.local_dir)
            repo.git_add(auto_lfs_track=True)
            repo.git_commit("Saved tokenizer")
            repo.git_push()
            logger.info(f"Tokenizer saved to repository at {self.config.hf_repo_id}")
        except Exception as e:
            logger.error(f"Error saving tokenizer to repository: {e}")

    def load_datasets(self) -> Dataset:
        datasets = []
        logger.info("Loading datasets...")
        for src_lang, tgt_lang in self.config.language_pairs:
            lang_pair = f"{src_lang}-{tgt_lang}"
            try:
                dataset = load_dataset(
                    self.config.dataset_name, lang_pair, split='train',
                    cache_dir=str(self.config.output_dir / "cache")
                )
                datasets.append(dataset)
                logger.info(f"Loaded dataset for language pair {lang_pair}.")
            except Exception as e:
                logger.error(f"Error loading dataset for {lang_pair}: {e}")
                raise DatasetLoadError(f"Error loading dataset for {lang_pair}: {str(e)}")

        if not datasets:
            raise DatasetLoadError("No datasets were loaded. Check your language pairs and dataset names.")

        combined_dataset = concatenate_datasets(datasets)
        logger.info("Datasets loaded and concatenated successfully.")
        return combined_dataset

    def preprocess_function(self, examples: Dict[str, Any], src_lang: str, tgt_lang: str) -> Dict[str, Any]:
        inputs = [ex[src_lang] for ex in examples["translation"]]
        targets = [ex[tgt_lang] for ex in examples["translation"]]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.max_input_length,
            truncation=True,
            padding="max_length"
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.config.max_target_length,
                truncation=True,
                padding="max_length"
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def process_data(self) -> DatasetDict:
        if not self.tokenizer:
            raise TokenizerNotInitializedError("Tokenizer not initialized. Call 'load_tokenizer' first.")

        dataset = self.load_datasets()
        processed_datasets = []

        logger.info("Processing datasets...")
        for src_lang, tgt_lang in self.config.language_pairs:
            lang_pair_dataset = dataset.filter(
                lambda example: src_lang in example["translation"] and tgt_lang in example["translation"],
                num_proc=self.config.num_proc
            )

            tokenized_dataset = lang_pair_dataset.map(
                lambda examples: self.preprocess_function(examples, src_lang, tgt_lang),
                batched=True,
                num_proc=self.config.num_proc,
                remove_columns=lang_pair_dataset.column_names
            )
            processed_datasets.append(tokenized_dataset)
            logger.info(f"Processed dataset for language pair {src_lang}-{tgt_lang}.")

        combined_tokenized_dataset = concatenate_datasets(processed_datasets)
        logger.info("Datasets processed successfully.")
        dataset_dict = combined_tokenized_dataset.train_test_split(test_size=0.1, seed=self.config.random_seed)
        logger.info("Datasets split into train and test sets.")
        return dataset_dict

# Translation Model class
class TranslationModel:
    def __init__(self, config: TranslationConfig, tokenized_datasets: DatasetDict, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenized_datasets = tokenized_datasets
        self.tokenizer = tokenizer
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self.trainer: Optional[Seq2SeqTrainer] = None
        self.accelerator: Optional[Accelerator] = None

    def load_model(self) -> None:
        logger.info("Loading model...")
        try:
            if self.config.use_8bit_quantization:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_checkpoint,
                    load_in_8bit=True,
                    device_map="auto"
                )
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_checkpoint)

            if self.config.use_peft:
                from peft import get_peft_model, LoraConfig, TaskType
                peft_config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    inference_mode=False,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout
                )
                self.model = get_peft_model(self.model, peft_config)
                logger.info("PEFT model configured with LoRA.")
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ModelNotInitializedError(f"Error loading model: {str(e)}")

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu = BLEU()
        result = bleu.corpus_score(decoded_preds, [decoded_labels])

        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = [rouge_scorer_obj.score(pred, label) for pred, label in zip(decoded_preds, decoded_labels)]
        rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
        rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])

        return {
            "bleu": result.score,
            "rouge1": rouge1,
            "rougeL": rougeL
        }

    def setup_trainer(self) -> None:
        logger.info("Setting up trainer...")
        self.accelerator = Accelerator(mixed_precision=self.config.mixed_precision)

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.config.output_dir),
            evaluation_strategy="steps",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            weight_decay=self.config.weight_decay,
            save_total_limit=3,
            num_train_epochs=self.config.num_train_epochs,
            predict_with_generate=True,
            fp16=self.config.mixed_precision == 'fp16',
            load_best_model_at_end=True,
            metric_for_best_model="eval_bleu",
            greater_is_better=True,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            max_grad_norm=self.config.max_grad_norm,
            logging_dir=str(self.config.output_dir / "logs"),
            logging_steps=100,
            report_to="wandb" if self.config.use_wandb else ["console"],
            dataloader_num_workers=self.config.num_proc
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["test"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)],
            compute_metrics=self.compute_metrics
        )
        logger.info("Trainer set up successfully.")

    def train_and_evaluate(self) -> Dict[str, float]:
        if self.config.use_wandb:
            wandb.init(project=self.config.wandb_project, config=self.config.__dict__)

        logger.info("Starting training...")
        train_result = self.trainer.train()
        logger.info("Training completed.")

        logger.info("Evaluating model...")
        metrics = self.trainer.evaluate()
        logger.info(f"Evaluation metrics: {metrics}")

        self.trainer.save_model()  # Saves the tokenizer too for Seq2Seq
        return metrics

    def translate(self, text: str, src_lang: Optional[str], tgt_lang: str) -> str:
        if not self.model:
            raise ModelNotInitializedError("Model not initialized. Call 'load_model' first.")
        if not self.accelerator:
            self.accelerator = Accelerator(mixed_precision=self.config.mixed_precision)

        self.model.eval()
        self.model.to(self.accelerator.device)
        try:
            if not src_lang:
                src_lang = detect(text)
                logger.info(f"Detected source language: {src_lang}")

            # Prepare the input text
            input_text = text.strip()
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self.config.max_input_length,
                truncation=True
            ).to(self.accelerator.device)

            # Generate the translation
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_target_length,
                num_beams=4,
                early_stopping=True
            )
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation
        except Exception as e:
            logger.error(f"Error during translation: {e}")
            raise TranslationError(f"Error during translation: {e}")

    def evaluate_on_test_set(self) -> Dict[str, float]:
        logger.info("Evaluating on test set...")
        metrics = self.trainer.evaluate(self.tokenized_datasets["test"])
        logger.info(f"Test set evaluation metrics: {metrics}")
        return metrics

    def push_to_hub(self) -> None:
        if not self.config.hf_repo_id or not self.config.hf_token:
            logger.warning("HF_REPO_ID or HF_TOKEN not provided. Skipping model upload.")
            return
        logger.info(f"Pushing model to Hugging Face Hub at {self.config.hf_repo_id}...")
        try:
            self.trainer.push_to_hub(commit_message="Model upload")
            logger.info("Model pushed to Hugging Face Hub successfully.")
        except Exception as e:
            logger.error(f"Error pushing model to Hugging Face Hub: {e}")

# Translation Pipeline class
class TranslationPipeline:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.translation_model: Optional[TranslationModel] = None

    def initialize(self) -> None:
        logger.info("Initializing pipeline...")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(self.config.random_seed)
        self.data_processor.load_tokenizer()
        if self.config.hf_repo_id and self.config.hf_token:
            self.data_processor.save_tokenizer_to_repo()

    def run(self) -> None:
        logger.info("Starting pipeline run...")
        tokenized_datasets = self.data_processor.process_data()
        self.translation_model = TranslationModel(
            self.config,
            tokenized_datasets,
            self.data_processor.tokenizer
        )
        self.translation_model.load_model()
        self.translation_model.setup_trainer()
        self.translation_model.train_and_evaluate()
        self.translation_model.evaluate_on_test_set()
        if self.config.hf_repo_id:
            self.translation_model.push_to_hub()
        logger.info("Pipeline run completed successfully.")

    def interactive_translation(self) -> None:
        if not self.translation_model:
            logger.error("Model not initialized. Please run 'initialize' and 'run' methods first.")
            return
        logger.info("Entering interactive translation mode.")
        available_languages = set([lang for pair in self.config.language_pairs for lang in pair])
        try:
            while True:
                src_lang = input(f"Enter source language ({'/'.join(available_languages)}), or 'q' to quit: ").strip()
                if src_lang.lower() == 'q':
                    break
                if src_lang not in available_languages:
                    print(f"Unsupported source language. Available languages: {available_languages}")
                    continue
                tgt_lang = input(f"Enter target language ({'/'.join(available_languages)}), or 'q' to quit: ").strip()
                if tgt_lang.lower() == 'q':
                    break
                if tgt_lang not in available_languages:
                    print(f"Unsupported target language. Available languages: {available_languages}")
                    continue
                text = input("Enter text to translate, or 'q' to quit: ").strip()
                if text.lower() == 'q':
                    break
                translation = self.translation_model.translate(text, src_lang, tgt_lang)
                print(f"Translation: {translation}")
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def batch_translation_from_file(self, input_file: str, output_file: str, src_lang: Optional[str], tgt_lang: str) -> None:
        if not self.translation_model:
            logger.error("Model not initialized. Please run 'initialize' and 'run' methods first.")
            return
        logger.info(f"Starting batch translation from {input_file} to {output_file}...")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                texts = f.readlines()
            translations = []
            for text in texts:
                translation = self.translation_model.translate(text.strip(), src_lang, tgt_lang)
                translations.append(translation)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(translations))
            logger.info(f"Batch translation completed. Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error during batch translation: {e}")

# Main entry point
def main():
    parser = argparse.ArgumentParser(description="Multilingual Translation Pipeline")
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH, help="Path to the configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "interactive", "batch"], default="train", help="Operation mode")
    parser.add_argument("--input_file", type=str, help="Input file for batch translation")
    parser.add_argument("--output_file", type=str, help="Output file for batch translation")
    parser.add_argument("--src_lang", type=str, help="Source language for translation", default=None)
    parser.add_argument("--tgt_lang", type=str, help="Target language for translation", default=None)
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, help="LoRA r parameter", default=None)
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha parameter", default=None)
    parser.add_argument("--lora_dropout", type=float, help="LoRA dropout parameter", default=None)
    # Additional parameters
    parser.add_argument("--use_wandb", action='store_true', help="Use Weights & Biases for logging")
    parser.add_argument("--num_train_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    args = parser.parse_args()

    try:
        # Load configuration from file
        if os.path.exists(args.config_path):
            with open(args.config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            config_dict = {}

        # Update config with CLI arguments
        cli_config = {k: v for k, v in vars(args).items() if v is not None}
        config_dict.update(cli_config)
        config = TranslationConfig(**config_dict)
        pipeline = TranslationPipeline(config)
        pipeline.initialize()
        if args.mode == "train":
            pipeline.run()
        elif args.mode == "interactive":
            pipeline.run()  # Ensure model is trained/loaded
            pipeline.interactive_translation()
        elif args.mode == "batch":
            if not all([args.input_file, args.output_file, args.tgt_lang]):
                raise ValueError("For batch mode, input_file, output_file, and tgt_lang must be provided")
            pipeline.run()  # Ensure model is trained/loaded
            pipeline.batch_translation_from_file(args.input_file, args.output_file, args.src_lang, args.tgt_lang)
        logger.info("Translation pipeline execution completed.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
