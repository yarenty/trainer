# LLM Fine-tuning for DataFusion Documentation

This project aims to fine-tune a Large Language Model (LLM) to understand and answer questions about DataFusion, Apache Arrow, and related Rust data processing libraries. The goal is to create a specialized LLM that can serve as an intelligent interface for querying documentation and codebase.

## Project Structure

- `src/`: Python source code for various utilities.
- `data/`: Stores processed documentation, Q&A pairs, and other data artifacts.
- `scripts/`: Contains Python scripts for data preparation, model training, and evaluation.
- `models/`: Will store fine-tuned LLM models and adapters.
- `tests/`: Unit and integration tests.
- `sources/`: Contains the raw source code repositories (e.g., DataFusion, Arrow, Ballista) used for documentation extraction.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install `uv` (if not already installed):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3.  **Create and activate a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

## Data Preparation

The `scripts/prepare_data.py` script is responsible for extracting documentation from the `sources/` directory, cleaning it, chunking it using various strategies, and generating question-answer pairs. It includes a placeholder for LLM-assisted Q&A generation using Ollama.

To prepare the data:

1.  **Ensure source repositories are in `sources/`:** Place the cloned repositories (e.g., `datafusion`, `arrow-rs`, `ballista`) into the `sources/` directory.

2.  **Run the data preparation script:**
    ```bash
    python scripts/prepare_data.py
    ```
    This will generate `*.jsonl` files in the `data/` directory, one for each processed repository (e.g., `data/datafusion_qa.jsonl`, `data/ballista_qa.jsonl`).

    **Note on Ollama Integration:**
    If you wish to use an actual LLM (like Llama 3.2) for Q&A generation during data preparation:
    -   Install Ollama: Follow instructions on [ollama.ai](https://ollama.ai/).
    -   Pull the desired model: `ollama pull llama3.2` (or your preferred Llama 3 variant).
    -   Install the Ollama Python library: `uv pip install ollama`.
    -   Uncomment the Ollama-related lines in `scripts/prepare_data.py`.

## Model Fine-tuning

The `scripts/train.py` script handles the fine-tuning of a base LLM (e.g., Llama 3.2) using the prepared Q&A datasets. It leverages PEFT (LoRA) and 4-bit quantization for efficient training.

**Prerequisites:**
-   A GPU with sufficient VRAM and compatible drivers.
-   Access to the chosen base model on Hugging Face (you might need to `huggingface-cli login`).

To fine-tune the model:

```bash
python scripts/train.py
```

## Model Deployment with Ollama

After fine-tuning, the `scripts/deploy_ollama.py` script prepares the model for deployment with Ollama. This involves merging LoRA adapters, converting to GGUF format, and creating an Ollama Modelfile.

**Prerequisites:**
-   **Clone `llama.cpp`:**
    ```bash
    git clone https://github.com/ggerganov/llama.cpp.git
    ```
-   **Build `llama.cpp`:**
    ```bash
    cd llama.cpp
    make
    ```
    (Ensure you have a C++ compiler like `g++` or `clang` installed.)
-   **Update `llama_cpp_path` in `scripts/deploy_ollama.py`:** Open the script and set the `llama_cpp_path` variable to the absolute path of your `llama.cpp` directory.

To deploy the model to Ollama:

1.  **Run the deployment script:**
    ```bash
    python scripts/deploy_ollama.py
    ```

2.  **Import the model into Ollama:** The script will output the exact `ollama create` command you need to run. It will look something like:
    ```
    ollama create datafusion-llama3 -f /path/to/your/models/ollama_ready/Modelfile
    ```

3.  **Run your fine-tuned model:**
    ```bash
    ollama run datafusion-llama3 "### Instruction:\nHow do I use the filter operation in DataFusion?\n\n### Response:"
    ```

## Testing and Evaluation (Coming Soon)

This section will cover how to test and evaluate the performance and knowledge transfer of the fine-tuned model.