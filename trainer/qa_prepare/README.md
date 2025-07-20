# Q&A Generation Module

A modular Python package for generating question-answer pairs from documentation and code files using LLM-based processing with robust fallback mechanisms. This is now a reusable Python module/package, not just a script.

## Overview

This package provides a clean, modular architecture for processing repositories and generating high-quality Q&A pairs for fine-tuning coding assistants. The code is organized into focused classes that handle specific responsibilities:

- **TextCleaner**: Handles text cleaning and preprocessing
- **Chunker**: Manages different text chunking strategies
- **LLM_QA**: Handles LLM interactions and Q&A generation with fallbacks
- **OutputConverter**: Manages output formatting and file operations
- **FileProcessor**: Coordinates the overall processing workflow

## Architecture

```
trainer/qa_prepare/
├── __init__.py           # Package initialization and exports
├── text_cleaner.py       # Text cleaning utilities
├── chunker.py            # Text chunking strategies
├── llm_qa.py             # LLM Q&A generation with fallbacks
├── output_converter.py   # Output formatting and file operations
├── file_processor.py     # Main processing coordination
└── README.md             # This file
```

## Installation & Import

Place the `trainer/qa_prepare` directory in your Python path or install as part of your project. Then import classes as follows:

```python
from trainer.qa_prepare import TextCleaner, Chunker, LLM_QA, OutputConverter, FileProcessor
```

## Classes

### TextCleaner

Handles cleaning and preprocessing of text content.

```python
from trainer.qa_prepare import TextCleaner

cleaner = TextCleaner()

# Clean general text
cleaned_text = cleaner.clean_text(raw_text)

# Clean code-specific text
cleaned_code = cleaner.clean_code_text(raw_code)

# Remove boilerplate
clean_text = cleaner.remove_boilerplate(text)
```

### Chunker

Manages different strategies for breaking text into manageable chunks.

```python
from trainer.qa_prepare import Chunker

chunker = Chunker()

# Chunk by headings
chunks = chunker.chunk_by_headings(text, min_chars=200)

# Chunk by paragraphs
chunks = chunker.chunk_by_paragraphs(text, min_chars=100)

# Fixed-size chunking
chunks = chunker.chunk_by_fixed_size(text, chunk_size=500, overlap=50)

# Code-specific chunking
chunks = chunker.chunk_code_by_blocks(code_text, min_chars=100)
```

### LLM_QA

Handles LLM interactions for Q&A generation with robust fallback mechanisms.

```python
from trainer.qa_prepare import LLM_QA
import ollama

client = ollama.Client()
llm_qa = LLM_QA(client, model_name="llama3.2")

# Generate Q&A pair
qa_pair = llm_qa.generate_qa_pair(text_chunk)
```

### OutputConverter

Manages output formatting, validation, and file operations.

```python
from trainer.qa_prepare import OutputConverter

converter = OutputConverter()

# Write Q&A pairs to JSONL
converter.write_qa_pairs_to_jsonl(qa_pairs, "output.jsonl")

# Validate Q&A pair
is_valid = converter.validate_qa_pair(qa_pair)

# Filter invalid pairs
valid_pairs = converter.filter_qa_pairs(all_pairs)

# Merge multiple JSONL files
converter.merge_jsonl_files(["file1.jsonl", "file2.jsonl"], "merged.jsonl")
```

### FileProcessor

Coordinates the overall processing workflow and manages concurrency.

```python
from trainer.qa_prepare import FileProcessor
import ollama

client = ollama.Client()
processor = FileProcessor(
    ollama_client=client,
    model_name="llama3.2",
    max_workers=8
)

# Process entire repository
summary = processor.process_repository(
    repo_name="my-repo",
    repo_path="/path/to/repo",
    output_base_dir="./output"
)
```

## Usage Examples

### Basic Usage

```python
from trainer.qa_prepare import FileProcessor
import ollama

# Initialize
client = ollama.Client()
processor = FileProcessor(client, model_name="llama3.2")

# Process a repository
summary = processor.process_repository(
    repo_name="example-repo",
    repo_path="/path/to/repo",
    output_base_dir="./output"
)

print(f"Generated {summary['total_qa_pairs']} Q&A pairs")
```

### Custom Processing

```python
from trainer.qa_prepare import TextCleaner, Chunker, LLM_QA, OutputConverter
import ollama

# Initialize components
client = ollama.Client()
cleaner = TextCleaner()
chunker = Chunker()
llm_qa = LLM_QA(client, "llama3.2")
converter = OutputConverter()

# Custom processing pipeline
raw_text = "..."
cleaned_text = cleaner.clean_text(raw_text)
chunks = chunker.chunk_by_headings(cleaned_text)

qa_pairs = []
for chunk in chunks:
    qa_pair = llm_qa.generate_qa_pair(chunk)
    if converter.validate_qa_pair(qa_pair):
        qa_pairs.append(qa_pair)

converter.write_qa_pairs_to_jsonl(qa_pairs, "custom_output.jsonl")
```

## Features

- **Modular Design**: Clean separation of concerns with focused classes
- **Robust Fallbacks**: Multiple fallback mechanisms for LLM failures
- **Concurrent Processing**: Efficient parallel processing of files
- **Flexible Chunking**: Multiple strategies for text chunking
- **Quality Validation**: Built-in validation and filtering of Q&A pairs
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Error Handling**: Graceful error handling throughout the pipeline

## Output Format

The package generates JSONL files with the following structure:

```json
{
  "question": "What does this function do?",
  "answer": "This function processes data by...",
  "source_file": "/path/to/file.py",
  "source_repo": "repository-name",
  "generated_at": "2024-01-01T12:00:00"
}
```

## Dependencies

- `ollama`: For LLM interactions
- `pathlib`: For file path handling
- `concurrent.futures`: For parallel processing
- Standard library modules: `json`, `logging`, `re`, `os`

## Error Handling

The package includes comprehensive error handling:

- **LLM Failures**: Automatic fallback to simpler Q&A generation
- **File Errors**: Graceful handling of file reading/writing errors
- **JSON Parsing**: Robust extraction of Q&A pairs from malformed LLM responses
- **Concurrent Processing**: Timeout handling and error isolation

## Performance

- **Concurrent Processing**: Configurable number of workers (default: 8)
- **Efficient Chunking**: Smart chunking strategies to minimize redundant processing
- **Memory Management**: Streaming processing of large files
- **Timeout Handling**: Prevents hanging on problematic files or LLM calls 