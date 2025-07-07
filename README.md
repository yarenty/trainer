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
-   **Clone `llama.cpp`:** You need the `llama.cpp` source code for the `convert.py` script. Clone it to `/opt/ml/trainer/llama.cpp` (as per previous steps).
    ```bash
    git clone https://github.com/ggerganov/llama.cpp.git /opt/ml/trainer/llama.cpp
    ```
-   **Build `llama.cpp`:** Navigate into the cloned `llama.cpp` directory and build it. This will create the `quantize` executable.
    ```bash
    cd /opt/ml/trainer/llama.cpp
    make
    ```
    (Ensure you have a C++ compiler like `g++` or `clang` installed.)
-   **Verify `llama_cpp_path` in `scripts/deploy_ollama.py`:** Ensure the `llama_cpp_path` variable in the script is set to `/opt/ml/trainer/llama.cpp`.

To deploy the model to Ollama:

1.  **Run the deployment script:**
    ```bash
    python scripts/deploy_ollama.py
    ```

2.  **Import the model into Ollama:** The script will output the exact `ollama create` command you need to run. It will look something like:
    ```bash
    ollama create datafusion-llama3 -f /path/to/your/models/ollama_ready/Modelfile
    ```

3.  **Run your fine-tuned model:**
    ```bash
    ollama run datafusion-llama3 "### Instruction:\nHow do I use the filter operation in DataFusion?\n\n### Response:"
    ```

## Testing and Evaluation

After successfully deploying your fine-tuned model to Ollama, you can begin testing its performance and knowledge of DataFusion. 

**Qualitative Testing:**

1.  **Interact via Ollama CLI:** Use the `ollama run` command to interact directly with your model. Ask questions related to DataFusion concepts, API usage, SQL syntax, and common operations. For example:
    ```bash
    ollama run datafusion-llama3
    >>> ### Instruction:\nExplain the purpose of the `DataFrame` API in DataFusion.\n\n### Response:
    ```
    Observe the model's responses. Do they accurately reflect the documentation? Are they concise and relevant?

2.  **Compare with Base Model:** If possible, compare the responses of your fine-tuned model with the original base Llama 3.2 model (or whichever base model you used). You should see an improvement in DataFusion-specific knowledge and a reduction in generic LLM responses.

3.  **Test Edge Cases and Ambiguities:** Pose questions that might be ambiguous or require a deeper understanding of DataFusion's nuances. For example, ask about differences between similar functions or how to handle specific error scenarios.

**Quantitative Evaluation (Advanced - Requires a Test Set):**

For a more rigorous evaluation, you would typically create a separate, unseen test dataset of DataFusion-related questions and their ground-truth answers. This dataset should *not* have been used during the fine-tuning process.

1.  **Create a Test Set:** Manually or semi-automatically generate a set of questions and expected answers covering various aspects of DataFusion.

2.  **Generate Model Responses:** Use a Python script to programmatically query your fine-tuned Ollama model with the questions from your test set and collect its responses.

3.  **Evaluate Metrics:** Use Natural Language Processing (NLP) evaluation metrics to compare the model's generated answers against the ground-truth answers. Relevant metrics include:
    *   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Measures the overlap of n-grams between the generated and reference summaries/answers. Useful for assessing content overlap.
    *   **BLEU (Bilingual Evaluation Understudy):** Measures the precision of n-grams. Often used for machine translation, but can be adapted for Q&A.
    *   **Semantic Similarity Metrics:** Libraries like `sentence-transformers` can be used to embed both the generated and reference answers into vector space and calculate cosine similarity, providing a measure of semantic closeness.

    This would involve writing a Python script that:
    -   Loads your test dataset.
    -   Uses the `ollama` Python client to send prompts to your local Ollama instance.
    -   Collects responses.
    -   Calculates and reports the chosen evaluation metrics.

By combining qualitative interaction and (optionally) quantitative metrics, you can assess how well your fine-tuned LLM has learned the DataFusion knowledge and is performing for your specific use case.
