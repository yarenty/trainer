# Q&A Data Quality Module

A modular Python package for enforcing data quality, format, and validation for Q/A datasets. Provides automated tools to prevent common pitfalls in Q/A data for LLM training and evaluation.

**Note:** This module is designed to automate the prevention steps outlined in [pitfalls_plan.md](../../pitfalls_plan.md), ensuring your data pipeline avoids the most common issues in LLM fine-tuning.

## Overview

This package provides a suite of utilities to automate and enforce data quality, formatting, and validation for Q/A datasets. Each class targets a specific aspect of data quality, such as format enforcement, deduplication, balance analysis, ambiguity flagging, code block validation, template compliance, output post-processing, and edge case sampling.

## Architecture

```
trainer/qa_data_quality/
├── __init__.py                # Package initialization and exports
├── format_enforcer.py         # Enforce Q/A template and end marker
├── deduplicator.py            # Deduplicate Q/A pairs
├── balance_analyzer.py        # Analyze topic/answer balance
├── ambiguity_flagger.py       # Flag ambiguous/multi-answer questions
├── code_block_validator.py    # Validate and auto-correct code blocks
├── prompt_template_checker.py # Check prompt/response template compliance
├── output_postprocessor.py    # Post-process and review model outputs
├── edge_case_sampler.py       # Sample edge cases for human review
└── README.md                  # This file
```

## Installation & Import

Place the `trainer/qa_data_quality` directory in your Python path or install as part of your project. Then import classes as follows:

```python
from trainer.qa_data_quality import (
    QAFormatEnforcer, QADeduplicator, QABalanceAnalyzer, QAAmbiguityFlagger,
    QACodeBlockValidator, QAPromptTemplateChecker, QAOutputPostProcessor, QAEedgeCaseSampler
)
```

## Classes

### QAFormatEnforcer
Enforces Q/A template and end marker in all .jsonl files under qa_data/*/*.jsonl.

```python
from trainer.qa_data_quality import QAFormatEnforcer
enforcer = QAFormatEnforcer()
report = enforcer.enforce_all_files()
print(report)
```

### QADeduplicator
Deduplicates Q/A pairs in all .jsonl files under qa_data/*/*.jsonl.

```python
from trainer.qa_data_quality import QADeduplicator
dedup = QADeduplicator()
report = dedup.deduplicate_all_files()
print(report)
```

### QABalanceAnalyzer
Analyzes topic distribution and answer length statistics in all .jsonl files.

```python
from trainer.qa_data_quality import QABalanceAnalyzer
analyzer = QABalanceAnalyzer()
report = analyzer.analyze_all_files()
print(report)
```

### QAAmbiguityFlagger
Flags ambiguous or multi-answer questions in all .jsonl files.

```python
from trainer.qa_data_quality import QAAmbiguityFlagger
flagger = QAAmbiguityFlagger()
report = flagger.flag_all_files()
print(report)
```

### QACodeBlockValidator
Validates and auto-corrects code blocks and formatting in all .jsonl files.

```python
from trainer.qa_data_quality import QACodeBlockValidator
validator = QACodeBlockValidator()
validation_report = validator.validate_all_files()
print(validation_report)
corrections = validator.auto_correct_all_files()
print(corrections)
```

### QAPromptTemplateChecker
Checks prompt/response template compliance in all .jsonl files.

```python
from trainer.qa_data_quality import QAPromptTemplateChecker
checker = QAPromptTemplateChecker()
report = checker.check_all_files()
print(report)
```

### QAOutputPostProcessor
Post-processes model outputs: trims at end marker, flags verbosity/length issues.

```python
from trainer.qa_data_quality import QAOutputPostProcessor
postproc = QAOutputPostProcessor()
report = postproc.process_all_files()
print(report)
```

### QAEedgeCaseSampler
Samples edge cases (longest/shortest answers, rarest topic, random) for human review.

```python
from trainer.qa_data_quality import QAEedgeCaseSampler
sampler = QAEedgeCaseSampler()
report = sampler.sample_all_files()
print(report)
```

## Features

- **Comprehensive Data Quality Checks**: Format, deduplication, ambiguity, code block, template, and more
- **Automated Correction**: Auto-corrects common formatting/code block issues
- **Balance & Ambiguity Analysis**: Detects topic imbalance and ambiguous questions
- **Edge Case Sampling**: Samples for human review and future LLM-based review
- **Extensible**: Designed for easy addition of new checks and review steps
- **Batch Processing**: Processes all .jsonl files in qa_data/*/

## Output/Report Format

Most methods return a dictionary report, e.g.:

```json
{
  "qa_data/datafusion/code_qa.jsonl": {
    "total_lines": 1000,
    "fixed_lines": 12,
    "issues": ["Line 5: Added end marker.", "Line 42: Missing question or answer."]
  },
  ...
}
```

## Dependencies

- `os`, `json`, `glob`, `logging`, `re`, `random`, `collections`, `statistics`
- `trainer.config` for DATA_DIR and END_MARKER

## Error Handling

- Handles malformed JSON lines gracefully
- Logs and reports issues per file and line
- Continues processing even if some files/lines have errors

## Performance

- Processes all files in batch mode
- Designed for efficient file I/O and reporting
- Suitable for large Q/A datasets 

## Planned Extensions

The following features are planned for future releases, as outlined in [pitfalls_plan.md](../../pitfalls_plan.md):

- **Data Split Overlap Checking:** Scripts to check for overlap between train/val/test splits and report statistics.
- **LLM-Based Review and Generation:** Use LLMs to review answer quality, flag or generate edge cases, and automate human-in-the-loop workflows.
- **Synthetic Validation/Test Generation:** Tools to generate synthetic validation/test questions using LLMs.
- **Automated Human Review Workflow:** Automate periodic human review (e.g., sample N outputs per epoch for review). 