## Project Plan: Fine-tuning LLM for DataFusion Knowledge

This document outlines the step-by-step plan for setting up a Python project, preparing DataFusion documentation data, fine-tuning a Llama 3.2 model, and integrating it with Ollama.

### Step 1: Project Setup with `uv`

-   **Objective**: Establish a clean and organized Python project environment.
-   **Actions Taken**:
    -   Created core directories: `src`, `data`, `scripts`, `models`, `tests`.
    -   Initialized `README.md`, `.gitignore`, and `pyproject.toml`.
    -   Set up a `uv` virtual environment (`uv venv`).
    -   Installed essential Python packages for data handling, LLM interaction, and fine-tuning (`pandas`, `scikit-learn`, `torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`, `datasets`, `fastparquet`, `pyarrow`).

### Step 2: Data Acquisition and Preparation

-   **Objective**: Extract, clean, and format DataFusion documentation into a suitable dataset for LLM fine-tuning.
-   **Data Sources**: Primary sources are the local copies of DataFusion documentation (e.g., `.rst`, `.md` files) found in `/opt/ml/trainer/sources/datafusion/docs/source/`.
-   **Process**:
    1.  **Extraction**: Read all `.rst` and `.md` files from the specified documentation directories.
    2.  **Cleaning**: Implement a `clean_text` function to remove reStructuredText/Markdown formatting, boilerplate, and excessive whitespace.
    3.  **Chunking Strategies**: Apply multiple methods to break down the cleaned text into manageable segments:
        *   **Heading-based**: Chunks content based on document headings, merging smaller sections to meet a minimum character count.
        *   **Paragraph-based**: Chunks by paragraphs (double newlines), merging small paragraphs.
        *   **Fixed-size with overlap**: Chunks into fixed-size segments with a configurable overlap, attempting to break at natural points (e.g., end of sentences).
    4.  **Q&A Generation (LLM-assisted)**: For each chunk, generate a concise question-answer pair. This step involves an LLM call.
        *   **Note on Ollama Integration**: The `scripts/prepare_data.py` script includes placeholder logic for LLM calls. To use an actual LLM (like Llama 3.2 via Ollama), you must:
            *   Install Ollama (`ollama.ai`).
            *   Pull the desired Llama 3.2 model (`ollama pull llama3.2`).
            *   Install the Ollama Python library (`uv pip install ollama`).
            *   Uncomment the Ollama-related lines in `scripts/prepare_data.py`.
    5.  **Storage**: Save the processed Q&A pairs as a JSONL file (`data/datafusion_qa.jsonl`). Each line will be a JSON object containing the question, answer, source chunk preview, and chunking strategy used.

### Step 3: Fine-tuning the LLM (Llama 3.2)

-   **Objective**: Adapt a pre-trained Llama 3.2 model to the DataFusion knowledge domain.
-   **Method**: Utilize Parameter-Efficient Fine-Tuning (PEFT), specifically LoRA (Low-Rank Adaptation), to efficiently train the model without modifying all parameters.
-   **Tools**: Hugging Face `transformers` for model loading, `peft` for LoRA, `bitsandbytes` for 4-bit quantization (memory efficiency), and `accelerate` for potential distributed training.
-   **Training Script (`scripts/train.py` - To be developed)**:
    1.  Load the `data/datafusion_qa.jsonl` dataset using the `datasets` library.
    2.  Load the Llama 3.2 base model and tokenizer.
    3.  Initialize `PeftModel` with LoRA configuration.
    4.  Configure `TrainingArguments` and `Trainer`.
    5.  Train the model, potentially leveraging quantization and acceleration.
    6.  Implement hyperparameter tuning (learning rate, batch size, epochs, LoRA parameters).

### Step 4: Model Storage and Ollama Integration

-   **Objective**: Prepare the fine-tuned model for deployment and serve it via Ollama.
-   **Process**:
    1.  **Save Adapters**: Store the trained LoRA adapters in the `models/` directory.
    2.  **Merge Adapters (Recommended)**: Merge the LoRA adapters with the base Llama 3.2 model to create a single, consolidated model for easier deployment.
    3.  **Quantization**: Quantize the merged model to a GGUF format for efficient inference with Ollama (e.g., using `llama.cpp` tools or Ollama's built-in capabilities).
    4.  **Ollama Modelfile**: Create a `Modelfile` that points to the quantized GGUF model and defines any necessary Ollama parameters.
    5.  **Import to Ollama**: Use `ollama create your_model_name -f Modelfile` to import the fine-tuned model into Ollama.

### Step 5: Testing and Evaluation

-   **Objective**: Verify the performance and knowledge transfer of the fine-tuned model.
-   **Methods**:
    1.  **Qualitative Evaluation**: Interact with the model via the Ollama interface, asking questions about DataFusion concepts, API usage, and operations. Compare responses against original documentation and the base Llama 3.2 model.
    2.  **Quantitative Evaluation (Optional)**: Develop a separate test set of DataFusion-specific questions and expected answers. Use metrics (e.g., ROUGE, BLEU, or custom semantic similarity) to automate evaluation of the model's generated responses.
