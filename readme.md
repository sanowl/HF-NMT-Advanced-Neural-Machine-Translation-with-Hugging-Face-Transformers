# Multilingual Translation Model Trainer

This repository contains a high-quality, multilingual translation model using the MT5-XL architecture. It handles data preparation, model training, evaluation, and inference for multiple language pairs.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training and Evaluation](#training-and-evaluation)
- [Inference](#inference)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Multilingual Support:** Train and evaluate translation models for multiple language pairs.
- **Data Processing:** Efficient data loading and preprocessing using the `datasets` library.
- **Model Training:** Fine-tune the MT5-XL model using Hugging Face's `transformers` library.
- **Evaluation:** Evaluate the model using BLEU scores and other relevant metrics.
- **Inference:** Translate text from one language to another using the trained model.
- **Logging:** Comprehensive logging for tracking progress and debugging.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To use the Multilingual Translation Model Trainer, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/sanowl/HF-NMT-Advanced-Neural-Machine-Translation-with-Hugging-Face-Transformers.git
    cd multilingual-translation-model-trainer
    ```

2. **Configure the settings:** Update the `TranslationConfig` in the `translation_trainer.py` file to match your requirements.

3. **Run the training pipeline:**

    ```bash
    python translation_trainer.py
    ```

## Configuration

The `TranslationConfig` class in `translation_trainer.py` allows you to configure various parameters for training:

- `random_seed`: Seed for reproducibility.
- `dataset_name`: The name of the dataset to be used.
- `language_pairs`: Tuples of source and target language pairs.
- `model_checkpoint`: Pre-trained model checkpoint from Hugging Face.
- `output_dir`: Directory to save the trained model and outputs.
- `num_proc`: Number of processes for data preprocessing.
- `batch_size`: Batch size for training.
- `learning_rate`: Learning rate for training.
- `num_train_epochs`: Number of training epochs.
- `max_input_length`: Maximum input sequence length.
- `max_target_length`: Maximum target sequence length.

## Training and Evaluation

The training pipeline involves loading datasets, preprocessing, setting up the model and trainer, and running the training and evaluation:

```python
pipeline.initialize()
pipeline.run()
```

The evaluation results, including BLEU scores, will be logged.

## Inference

To translate text using the trained model:

```python
translated_text = translation_model.translate("Your text here", target_lang="fr")
print(f"Translated Text: {translated_text}")
```

## Logging

Logging is set up using the `logging` module to provide detailed information about the training process:

```python
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Push your branch and create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
