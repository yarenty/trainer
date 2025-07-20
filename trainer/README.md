# Trainer: Modular LLM Pipeline Framework

A modular Python package for end-to-end large language model (LLM) data preparation, training, evaluation, and deployment. The `trainer` package provides a complete, auditable workflow for building, fine-tuning, and validating LLMs, with a focus on code and Q/A tasks.

---

## Overview

The `trainer` package is organized into focused submodules, each responsible for a key part of the LLM pipeline:

- **`qa_prepare`**: Modular Q&A pair generation from code and documentation using LLMs.
- **`qa_data_quality`**: Automated data quality checks, validation, and cleaning for Q/A datasets.
- **`steps`**: Step-by-step scripts for model download, training, conversion, quantization, evaluation, and deployment.
- **`utils`**: Shared utility functions for data loading, logging, and model training.

This structure enables reproducible, high-quality LLM workflows, from raw data to deployable models.

---

## Architecture

```
trainer/
├── __init__.py
├── config.py                # Central configuration
├── main.py                  # (Optional) Main entry point
├── prepare_data.py          # Data preparation script
├── post_processing.py       # Output post-processing
├── qa_prepare/              # Q&A generation module
├── qa_data_quality/         # Data quality and validation module
├── steps/                   # Modular pipeline step scripts
├── utils/                   # Shared utility functions
```

---

## Submodules

### [qa_prepare](./qa_prepare/README.md)
- **Purpose:** Generate high-quality Q&A pairs from code and documentation using LLMs.
- **Features:** Modular design, robust fallback, concurrent processing, output validation.

### [qa_data_quality](./qa_data_quality/README.md)
- **Purpose:** Enforce data quality, format, and validation for Q/A datasets.
- **Features:** Format enforcement, deduplication, balance analysis, ambiguity flagging, code block validation, template compliance, output post-processing, edge case sampling.

### [steps](./steps/README.md)
- **Purpose:** Modular scripts for each step in the LLM fine-tuning and deployment workflow.
- **Features:** Download, train, merge, convert, quantize, package, evaluate, and upload models.

### [utils](./utils/README.md)
- **Purpose:** Shared utility functions for model training, data loading, logging, and adapter-based fine-tuning.
- **Features:** Data tokenization, logging setup, CPU/GPU/LoRA/SFT training helpers.

---

## Example Workflow

1. **Prepare Q&A Data:**  
   Use `qa_prepare` to generate Q&A pairs from your codebase and documentation.
2. **Validate and Clean Data:**  
   Run `qa_data_quality` checks to ensure data quality and consistency.
3. **Run Pipeline Steps:**  
   Use the scripts in `steps` to download, train, convert, quantize, and deploy your model.
4. **Leverage Utilities:**  
   Use `utils` for data loading, logging, and custom training workflows.

---

## Features

- **End-to-End Pipeline:** From raw data to deployable LLMs.
- **Modular and Auditable:** Each step and utility is independently runnable and testable.
- **Quality-First:** Built-in data validation and cleaning.
- **Flexible:** Supports multiple training modes and deployment targets (Hugging Face, Ollama, llama.cpp).

---

## Dependencies

- `transformers`, `datasets`, `trl`, `peft`, `huggingface_hub`, `torch`
- Standard Python libraries: `os`, `json`, `glob`, `logging`, `subprocess`, `shutil`

---

## Getting Started

1. Clone the repository and install dependencies.
2. Configure your settings in `trainer/config.py`.
3. Follow the example workflow above, or see each submodule’s README for details.

---

## Planned Extensions

- Full pipeline orchestration and automation.
- CI/CD integration for continuous model updates.
- Additional data augmentation and validation tools.
- Expanded support for new model architectures and adapters.

---

For detailed usage and API documentation, see the README in each submodule. 