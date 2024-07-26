import logging, os, torch, numpy as np, json, wandb
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from datasets import concatenate_datasets, load_dataset, DatasetDict, Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, Repository
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments
from sacrebleu.metrics import BLEU
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup

load_dotenv()

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
    use_ray: bool = False
    num_gpu: int = 1
    use_multi_gpu: bool = False
    log_level: str = "INFO"
    early_stopping_patience: int = 3
    scheduler_type: str = "linear"
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    clipping_threshold: float = 1.0
    use_fsdp: bool = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranslationError(Exception): pass
class ModelNotInitializedError(TranslationError): pass
class TokenizerNotInitializedError(TranslationError): pass
class DatasetLoadError(TranslationError): pass

class DataProcessor:
    def __init__(self, config: TranslationConfig): self.config, self._tokenizer = config, None
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
        for dataset in datasets: dataset = dataset.map(self._align_schema)
        return concatenate_datasets(datasets)
    def _align_schema(self, example: Dict[str, Any]) -> Dict[str, Any]: return {"translation": example.get(f"translation_{src_lang}_{tgt_lang}") for src_lang, tgt_lang in self.config.language_pairs}
    def preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        inputs = [examples["translation"][src_lang] for src_lang, _ in self.config.language_pairs]
        targets = [examples["translation"][tgt_lang] for _, tgt_lang in self.config.language_pairs]
        model_inputs = self._tokenizer(inputs, max_length=self.config.max_input_length, truncation=True, padding="max_length")
        labels = self._tokenizer(targets, max_length=self.config.max_target_length, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    def load_tokenizer(self) -> None: self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_checkpoint)
    def process_data(self) -> DatasetDict:
        self.load_tokenizer()
        dataset = self.load_datasets()
        tokenized_datasets = dataset.map(self.preprocess_function, batched=True, num_proc=self.config.num_proc, remove_columns=dataset.column_names)
        return tokenized_datasets.train_test_split(test_size=0.1)
    def save_tokenizer_to_repo(self) -> None:
        logger.info(f"Saving tokenizer to repository at {self.config.hf_repo_id}...")
        repo = Repository(local_dir=str(self.config.output_dir), clone_from=f"https://huggingface.co/{self.config.hf_repo_id}", use_auth_token=self.config.hf_token)
        self._tokenizer.save_pretrained(repo.local_dir)
        repo.git_add(auto_lfs_track=True)
        repo.git_commit("Saved tokenizer")
        repo.git_push()
        logger.info(f"Tokenizer saved to repository at {self.config.hf_repo_id}")

class TranslationModel:
    def __init__(self, config: TranslationConfig, tokenized_datasets: DatasetDict, tokenizer: AutoTokenizer):
        self.config, self.tokenized_datasets, self._tokenizer = config, tokenized_datasets, tokenizer
        self._model, self._trainer, self.accelerator = None, None, None
    def load_model(self) -> None:
        if self.config.use_8bit_quantization: self._model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_checkpoint, load_in_8bit=True, device_map="auto")
        else: self._model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_checkpoint)
        if self.config.use_peft:
            from peft import get_peft_model, LoraConfig, TaskType
            peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=self.config.lora_r, lora_alpha=self.config.lora_alpha, lora_dropout=self.config.lora_dropout)
            self._model = get_peft_model(self._model, peft_config)
            self._model.print_trainable_parameters()
    def setup_trainer(self) -> None:
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.config.output_dir), evaluation_strategy="steps", learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size, per_device_eval_batch_size=self.config.batch_size,
            weight_decay=self.config.weight_decay, save_total_limit=3, num_train_epochs=self.config.num_train_epochs,
            predict_with_generate=True, fp16=self.config.use_fp16, load_best_model_at_end=True,
            metric_for_best_model="eval_bleu", gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps, eval_steps=self.config.eval_steps, save_steps=self.config.save_steps,
            max_grad_norm=self.config.max_grad_norm, logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=100, report_to="wandb" if self.config.use_wandb else None
        )
        data_collator = DataCollatorForSeq2Seq(self._tokenizer, model=self._model)
        optimizer = AdamW(self._model.parameters(), lr=self.config.learning_rate, eps=self.config.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=len(self.tokenized_datasets["train"]) * self.config.num_train_epochs)
        self._trainer = Seq2SeqTrainer(
            model=self._model, args=training_args, train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["test"], data_collator=data_collator, tokenizer=self._tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)],
            compute_metrics=self.compute_metrics, optimizers=(optimizer, scheduler)
        )
        if self.config.use_accelerate:
            self.accelerator = Accelerator()
            self._model, self._trainer.optimizer, self._trainer.lr_scheduler, self._trainer.train_dataloader = self.accelerator.prepare(
                self._model, self._trainer.optimizer, self._trainer.lr_scheduler, self._trainer.train_dataloader
            )
        if self.config.use_deepspeed: self._trainer.accelerator.state.deepspeed_plugin.deepspeed_config = json.load(open(self.config.deepspeed_config_path))
        if self.config.use_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
            from torch.distributed.fsdp.wrap import enable_wrap, wrap
            self._model = FSDP(self._model, cpu_offload=CPUOffload(offload_params=True))
    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple): preds = preds[0]
        decoded_preds = self._tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self._tokenizer.pad_token_id)
        decoded_labels = self._tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu = BLEU()
        result = bleu.corpus_score(decoded_preds, [decoded_labels])
        prediction_lens = [np.count_nonzero(pred != self._tokenizer.pad_token_id) for pred in preds]
        result = {"bleu": result.score, "length": np.mean(prediction_lens)}
        return result
    def train_and_evaluate(self) -> Dict[str, float]:
        if self.config.use_wandb: wandb.init(project=self.config.wandb_project, config=self.config.__dict__)
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
        api.upload_folder(folder_path=str(self.config.output_dir / "best_model"), path_in_repo="", repo_id=self.config.hf_repo_id, token=self.config.hf_token)
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
    def get_model_size(self) -> int: return sum(p.numel() for p in self._model.parameters())
    def get_gpu_memory_usage(self) -> float: return torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    def log_training_info(self) -> None:
        logger.info(f"Model size: {self.get_model_size()} parameters")
        logger.info(f"GPU memory usage: {self.get_gpu_memory_usage():.2f} GB")
        logger.info(f"Dataset size: {len(self.tokenized_datasets['train'])} training samples, {len(self.tokenized_datasets['test'])} test samples")
    def run_inference(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]: return [self.translate(text, src_lang, tgt_lang) for text in texts]

class TranslationPipeline:
    def __init__(self, config: TranslationConfig): self.config = config; self.data_processor = None; self.translation_model = None
    def initialize(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(self.config.random_seed)
        self.data_processor = DataProcessor(self.config)
        self.data_processor.load_tokenizer()
        self.data_processor.save_tokenizer_to_repo()
    def run(self) -> None:
        tokenized_datasets = self.data_processor.process_data()
        logger.info("Dataset processing completed.")
        self.translation_model = TranslationModel(self.config, tokenized_datasets, self.data_processor._tokenizer)
        self.translation_model.load_model()
        self.translation_model.setup_trainer()
        logger.info("Starting model training and evaluation...")
        train_results = self.translation_model.train_and_evaluate()
        logger.info(f"Training completed. Results: {train_results}")
        logger.info("Evaluating model on test set...")
        test_results = self.translation_model.evaluate_on_test_set(tokenized_datasets["test"])
        logger.info(f"Test set evaluation results: {test_results}")
        self.translation_model.save_model()
        if self.config.hf_repo_id: self.translation_model.push_to_hub()
        self.translation_model.log_training_info()
    def translate_batch(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]: return self.translation_model.run_inference(texts, src_lang, tgt_lang)
    def interactive_translation(self) -> None:
        print("Interactive Translation Mode")
        print("Enter 'q' to quit")
        while True:
            src_lang = input("Enter source language: ").strip()
            if src_lang.lower() == 'q': break
            tgt_lang = input("Enter target language: ").strip()
            if tgt_lang.lower() == 'q': break
            text = input("Enter text to translate: ").strip()
            if text.lower() == 'q': break
            translation = self.translation_model.translate(text, src_lang, tgt_lang)
            print(f"Translation: {translation}")
    def batch_translation_from_file(self, input_file: str, output_file: str, src_lang: str, tgt_lang: str) -> None:
        with open(input_file, 'r') as f: texts = f.readlines()
        translations = self.translate_batch([text.strip() for text in texts], src_lang, tgt_lang)
        with open(output_file, 'w') as f: f.write('\n'.join(translations))
        logger.info(f"Batch translation completed. Results saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multilingual Translation Pipeline")
    parser.add_argument("--config_path", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "interactive", "batch"], default="train", help="Operation mode")
    parser.add_argument("--input_file", type=str, help="Input file for batch translation")
    parser.add_argument("--output_file", type=str, help="Output file for batch translation")
    parser.add_argument("--src_lang", type=str, help="Source language for translation")
    parser.add_argument("--tgt_lang", type=str, help="Target language for translation")
    args = parser.parse_args()
    with open(args.config_path, 'r') as f: config_dict = json.load(f)
    config = TranslationConfig(**config_dict)
    pipeline = TranslationPipeline(config)
    pipeline.initialize()
    if args.mode == "train": pipeline.run()
    elif args.mode == "interactive": pipeline.interactive_translation()
    elif args.mode == "batch":
        if not all([args.input_file, args.output_file, args.src_lang, args.tgt_lang]):
            raise ValueError("For batch mode, input_file, output_file, src_lang, and tgt_lang must be provided")
        pipeline.batch_translation_from_file(args.input_file, args.output_file, args.src_lang, args.tgt_lang)
    logger.info("Translation pipeline execution completed.")
