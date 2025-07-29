# Utility Functions for LLM Training (`trainer.utils`)

A collection of utility modules supporting model training, data processing, logging, and adapter-based fine-tuning for LLM workflows. These utilities are used throughout the training pipeline to provide reusable, modular functionality.

---

## Overview

This package provides shared functions and helpers for model training (CPU, GPU, LoRA, SFT), data loading and tokenization, and logging setup. It is designed to be imported and used by the main pipeline steps and scripts.

---

## Architecture

```
trainer/utils/
├── __init__.py         # Module docstring
├── data_utils.py       # Data loading and tokenization utilities
├── logging_utils.py    # Logging setup utility
├── train_cpu.py        # Standard CPU-based training
├── train_gpu.py        # Standard GPU-based training
├── train_lora.py       # LoRA/PEFT adapter training
├── train_sft.py        # SFTTrainer-based training
```

---

## Usage

Import the relevant utility in your training or pipeline scripts:

```python
from trainer.utils.data_utils import load_and_tokenize_dataset
from trainer.utils.logging_utils import setup_logging
from trainer.utils.train_cpu import train_cpu
from trainer.utils.train_gpu import train_gpu
from trainer.utils.train_lora import train_lora
from trainer.utils.train_sft import train_sft
```

---

## Utility Descriptions

### data_utils.py
**Shared utilities for data loading and tokenization.**
- `load_and_tokenize_dataset(data_dir, tokenizer, max_length=2048)`: Loads and tokenizes a dataset from JSONL files for use in training.

### logging_utils.py
**Project-wide logging setup.**
- `setup_logging(verbose=False)`: Configures logging format and level.

### train_cpu.py
**Standard CPU-based model training.**
- `train_cpu(model_dir, data_dir, output_dir, config)`: Trains a model on CPU using Hugging Face Trainer.

### train_gpu.py
**Standard GPU-based model training.**
- `train_gpu(model_dir, data_dir, output_dir, config)`: Trains a model on GPU with mixed precision and larger batch size.

### train_lora.py
**LoRA/PEFT adapter-based training.**
- `train_lora(model_dir, data_dir, output_dir, config, device='cpu')`: Trains a model using LoRA/PEFT adapters.

### train_sft.py
**SFTTrainer-based training (using TRL).**
- `train_sft(model_dir, data_dir, output_dir, config, device='cpu')`: Trains a model using the TRL SFTTrainer.

---

## Features

- **Reusable Utilities:** Shared functions for data, logging, and training.
- **Supports Multiple Training Modes:** CPU, GPU, LoRA/PEFT, and SFT.
- **Consistent Data Handling:** Centralized data loading and tokenization.
- **Easy Logging Configuration:** Standardized logging for all scripts.

---

## Dependencies

- `transformers`
- `datasets`
- `trl` (for SFTTrainer)
- `peft` (for LoRA/PEFT)
- `torch`
- `os`, `glob`, `json`, `logging`

---

## Planned Extensions

- Add more data preprocessing and augmentation utilities.
- Expand logging utilities for advanced monitoring.
- Add support for additional adapter/fine-tuning methods. 