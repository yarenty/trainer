# LLM Training Pipeline Steps Module

A modular collection of Python scripts for each step in the LLM fine-tuning and deployment workflow. Each script is designed to be run independently, enabling a clear, reproducible, and auditable pipeline from model download to evaluation and upload.

**Note:** This module provides the step-by-step automation for the LLM workflow described in project’s documentation.

---

## Overview

This package contains scripts for each major step in the LLM fine-tuning pipeline, including model download, training, merging, conversion, quantization, packaging for Ollama, evaluation, and uploading to Hugging Face. Each script can be run as a standalone module or integrated into a larger workflow.

---

## Architecture

```
trainer/steps/
├── __init__.py         # Module docstring
├── 1_download.py       # Download base model from Hugging Face
├── 2_train.py          # Train/fine-tune the model
├── 3_merge.py          # Merge fine-tuned model
├── 4_gguf.py           # Convert to GGUF format
├── 5_quantize.py       # Quantize GGUF model
├── 6_ollama.py         # Create Ollama Modelfile and import instructions
├── 7_evaluate.py       # Evaluate quantized model
├── 8_upload_to_hf.py   # Upload model to Hugging Face Hub
```

---

## Usage

Each script is intended to be run in order, but can also be run independently as needed. Example:

```bash
python -m trainer.steps.1_download
python -m trainer.steps.2_train
python -m trainer.steps.3_merge
python -m trainer.steps.4_gguf
python -m trainer.steps.5_quantize
python -m trainer.steps.6_ollama
python -m trainer.steps.7_evaluate
python -m trainer.steps.8_upload_to_hf
```

---

## Step Descriptions

### 1. 1_download.py
**Downloads the base model and tokenizer from Hugging Face.**  
- Uses `transformers` to fetch and save the model locally.
- Configurable via `trainer/config.py`.

### 2. 2_train.py
**Trains or fine-tunes the model.**  
- Supports CPU/GPU and multiple training modes (full, LoRA, SFT).
- Uses utility functions and configuration from `trainer/config.py`.

### 3. 3_merge.py
**Merges the fine-tuned model.**  
- Copies the fine-tuned model directory to a merged model directory.
- Prepares for conversion to GGUF.

### 4. 4_gguf.py
**Converts the merged model to GGUF format.**  
- Uses `llama.cpp`'s `convert_hf_to_gguf.py` script.
- Ensures compatibility with GGUF-based inference.

### 5. 5_quantize.py
**Quantizes the GGUF model.**  
- Uses `llama.cpp`'s `llama-quantize` tool.
- Supports configurable quantization types.

### 6. 6_ollama.py
**Creates a Modelfile for Ollama and prints import instructions.**  
- Prepares the quantized model for use with Ollama.
- Prints commands for importing and running the model.

### 7. 7_evaluate.py
**Provides instructions for evaluating the quantized model.**  
- Loads test questions from the QA dataset.
- Prints example commands for evaluation with Ollama and llama.cpp.

### 8. 8_upload_to_hf.py
**Uploads the model and tokenizer to Hugging Face Hub.**  
- Uses `huggingface_hub` and `transformers` to push model assets.
- Optionally uploads GGUF files as assets.

---

## Features

- **Modular Pipeline:** Each step is a standalone script, enabling flexible and auditable workflows.
- **Configurable:** Uses `trainer/config.py` for all paths and settings.
- **Compatible with Hugging Face, llama.cpp, and Ollama.**
- **Logging and Error Handling:** Each script logs progress and errors for traceability.

---

## Dependencies

- `transformers`
- `huggingface_hub`
- `llama.cpp` (for GGUF conversion and quantization)
- `os`, `shutil`, `subprocess`, `logging`, `json`, `glob`

---

## Planned Extensions

- Add orchestration scripts to run the full pipeline end-to-end.
- Integrate with CI/CD for automated model updates.
- Add more robust error handling and reporting.
- Support for additional quantization and conversion formats. 