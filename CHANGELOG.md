# Changelog

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