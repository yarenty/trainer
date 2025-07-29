# LLM Training Pipeline

This project provides a **modular, extensible pipeline** for training, fine-tuning, merging, converting, quantizing, and evaluating large language models (LLMs). It supports:

- **Full fine-tuning** (standard Hugging Face Trainer)
- **LoRA/PEFT** parameter-efficient fine-tuning
- **SFTTrainer** (trl) supervised fine-tuning
- GGUF conversion and quantization for llama.cpp/Ollama
- Ollama Modelfile creation and import
- Evaluation and compatibility checks

## Project Structure

```
.
├── trainer/
│   ├── qa_prepare/         # Q&A generation from code/docs ([README](trainer/qa_prepare/README.md))
│   ├── qa_data_quality/    # Data quality checks and cleaning ([README](trainer/qa_data_quality/README.md))
│   ├── steps/              # Modular pipeline step scripts ([README](trainer/steps/README.md))
│   ├── utils/              # Shared utility functions ([README](trainer/utils/README.md))
│   ├── config.py           # Central configuration
│   ├── main.py             # (Optional) Main CLI entry point
│   ├── prepare_data.py     # Data preparation script
│   ├── post_processing.py  # Output post-processing
│   └── ...                 # Other helpers
├── qa_data/                # Prepared Q&A data
├── sources/                # Source code repositories for data prep
└── ...
```

## Modules Overview

- **[trainer/qa_prepare](trainer/qa_prepare/README.md):** Generate Q&A pairs from code and documentation using LLMs.
- **[trainer/qa_data_quality](trainer/qa_data_quality/README.md):** Enforce data quality, deduplication, and formatting for Q/A datasets.
- **[trainer/steps](trainer/steps/README.md):** Modular scripts for each pipeline step (download, train, merge, convert, quantize, evaluate, upload).
- **[trainer/utils](trainer/utils/README.md):** Shared utilities for data loading, logging, and training.

See each module’s README for detailed usage and API documentation.

## Features
- Step-by-step scripts for each stage of the pipeline (see `trainer/steps/`)
- CLI with `--train-mode` (`full`, `lora`, `sft`) and `--device` (`cpu`, `gpu`)
- All configuration (paths, model names, defaults) in `trainer/config.py`
- Utilities for each training mode and shared data processing
- Google-style docstrings, logging, and robust error handling
- Follows strict coding conventions for readability and maintainability

## Installation

1. **Clone this repository:**
   ```sh
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Install additional tools:**
   - [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):
     ```sh
     pip install huggingface_hub
     huggingface-cli login
     ```
   - [trl (for SFTTrainer)](https://github.com/huggingface/trl):
     ```sh
     pip install trl
     ```
   - [peft (for LoRA/PEFT)](https://github.com/huggingface/peft):
     ```sh
     pip install peft
     ```
   - [datasets](https://github.com/huggingface/datasets):
     ```sh
     pip install datasets
     ```

5. **Install and build llama.cpp:**
   ```sh
   git clone https://github.com/ggerganov/llama.cpp.git llama.cpp
   cd llama.cpp
   cmake .
   make
   
   #yep.. any cuda did not work ... but re-check in future 
   cmake -B build
   cmake --build build --config Release 
   cd ..
   ```
   - This will build the necessary GGUF conversion and quantization tools.

6. **Ollama Setup**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Test installation
ollama --version
```

## Download Source Repositories

Place the source code repositories you want to use for data preparation in the `sources/` directory. For example:

```sh
cd sources
# Example: DataFusion, Arrow, Ballista
git clone https://github.com/apache/arrow-datafusion.git datafusion
git clone https://github.com/apache/arrow-rs.git arrow-rs
# Add more as needed
```

## Data Preparation

1. **Prepare your data:**
   - Use or adapt the provided scripts to extract documentation, code, and generate Q&A pairs from the `sources/` directory.
   - Place the resulting `.jsonl` files in the `qa_data/` directory.
   - Each `.jsonl` file should contain objects with `question` and `answer` fields.

2. **Example data preparation script:**
   - (You may need to write or adapt a script for your specific sources. See `train/prepare_data.py` for examples.)

**Tip:** Always run data quality checks (`trainer/qa_data_quality`) on your Q&A data before training to ensure high-quality results.

## Pipeline Usage

You can run the pipeline using the main CLI or by executing individual step scripts for more control:

**Using the CLI:**
```sh
python -m trainer.main --step 1_download
python -m trainer.main --step 2_train --train-mode lora --device gpu
python -m trainer.main --step 3_merge
python -m trainer.main --step 4_gguf
python -m trainer.main --step 5_quantize
python -m trainer.main --step 6_ollama
python -m trainer.main --step 7_evaluate
```

**Or run individual step scripts:**
```sh
python -m trainer.steps.1_download
python -m trainer.steps.2_train
python -m trainer.steps.3_merge
python -m trainer.steps.4_gguf
python -m trainer.steps.5_quantize
python -m trainer.steps.6_ollama
python -m trainer.steps.7_evaluate
```

- If you omit `--train-mode` or `--device`, the defaults from `config.py` are used.

## Switch Training Modes Easily
- Use `--train-mode full`, `--train-mode lora`, or `--train-mode sft` to select your preferred fine-tuning strategy.

## Customize and Extend
- Add new steps, utilities, or models as needed. The pipeline is designed for easy extension and experimentation.

## Workflow Overview

1. **Download**: Fetch base model and tokenizer from Hugging Face.
2. **Train**: Fine-tune using full, LoRA, or SFTTrainer mode.
3. **Merge**: Merge adapters (if needed) or copy fine-tuned model.
4. **GGUF Conversion**: Convert to GGUF format for llama.cpp/Ollama.
5. **Quantize**: Quantize the GGUF model for efficient inference.
6. **Ollama Import**: Create Modelfile and import into Ollama.
7. **Evaluate**: Test model on Datafusion QA tasks.

## Troubleshooting

### Common Issues:

- CUDA Out of Memory: Reduce batch size, use gradient checkpointing
- Slow Training: Enable mixed precision (fp16), use gradient accumulation
- Poor Performance: Increase dataset size, adjust learning rate
- Model Too Large: Use smaller base model or more aggressive quantization

### Performance Tips:

- Use unsloth for fastest training
- Enable gradient_checkpointing to save memory
- Use load_in_4bit=True for smaller models
- Monitor GPU utilization with nvidia-smi

## Planned Extensions

- Full pipeline orchestration and automation
- CI/CD integration for continuous model updates
- Additional data augmentation and validation tools
- Expanded support for new model architectures and adapters

## License
See `LICENSE` for details.



![Alt](https://repobeats.axiom.co/api/embed/5ab1b4862ec4693656a0a125e3b2c14b95ccb973.svg "Repobeats analytics image")

