# Changelog

## 2025-07-15
### Added
- Modular pipeline with step-by-step scripts for LLM training, merging, conversion, quantization, Ollama import, and evaluation.
- Support for three training modes: full fine-tuning, LoRA/PEFT, and SFTTrainer (trl).
- CLI now supports --train-mode and --device, with defaults set in config.py.
- Utilities for each training mode and shared data processing.
- Config-driven workflow: all paths, model names, and defaults are set in config.py.
- Google-style docstrings, logging, and error handling throughout.

### Notes
- This is now a full LLM training pipeline supporting both parameter-efficient and full fine-tuning, with modular, extensible design.

## 2025-07-09
### Added
- Major modularization and refactoring of data preparation pipeline:
  - Split logic into `TextCleaner`, `Chunker`, `LLM_QA`, `OutputConverter`, and `FileProcessor` classes
  - Thread-safe, incremental output: Q&A pairs are written to disk immediately as they are generated
  - Robust Q&A extraction logic (handles multi-line, code, YAML, and special characters)
  - Improved logging and error handling
  - Ready for large-scale, concurrent processing

## 2025-07-07
### Added
- Initial version with simple data preparation approach:
  - Sequential processing of files
  - Basic Q&A extraction and output
  - Minimal error handling and logging 