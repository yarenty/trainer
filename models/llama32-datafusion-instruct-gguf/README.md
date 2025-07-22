---
license: meta-llama/llama-3-2
base_model: yarenty/llama32-datafusion-instruct
tags:
- text-generation
- instruction
- datafusion
- rust
- code
- gguf
---

# Llama 3.2 DataFusion Instruct (GGUF)

This repository contains the GGUF version of the `yarenty/llama32-datafusion-instruct` model, quantized for efficient inference on CPU and other compatible hardware.

For full details on the model, including its training procedure, data, intended use, and limitations, please see the **[full model card](https://huggingface.co/yarenty/llama32-datafusion-instruct)**.

## Model Details

- **Base model:** [yarenty/llama32-datafusion-instruct](https://huggingface.co/yarenty/llama32-datafusion-instruct)
- **Format:** GGUF
- **Quantization:** `Q4_K_M` (Please verify and change if different)

## Prompt Template

This model follows the same instruction prompt template as the base model:

```
### Instruction:
{Your question or instruction here}

### Response:
```

## Usage

These files are compatible with tools like `llama.cpp` and `Ollama`.

### With Ollama

1.  Create the `Modelfile`:
    ```
    FROM ./llama32_datafusion.gguf
    TEMPLATE """### Instruction:
    {{ .Prompt }}

    ### Response:
    """
    PARAMETER stop "### Instruction:"
    PARAMETER stop "### Response:"
    PARAMETER stop "### End"
    ```

2.  Create and run the Ollama model:
    ```bash
    ollama create llama32-datafusion-instruct-gguf -f Modelfile
    ollama run llama32-datafusion-instruct-gguf "How do I use the Ballista scheduler?"
    ```

### With llama.cpp

```bash
./main -m llama32_datafusion.gguf --color -p "### Instruction:\nHow do I use the Ballista scheduler?\n\n### Response:" -n 256 --stop "### Instruction:" --stop "### Response:" --stop "### End"
```

## Citation

If you use this model, please cite the original base model:
```
@misc{yarenty_2025_llama32_datafusion_instruct,
  author = {yarenty},
  title = {Llama 3.2 DataFusion Instruct},
  year = {2025},
  publisher = {Hugging Face},
  journal = {Hugging Face repository},
  howpublished = {\url{https://huggingface.co/yarenty/llama32-datafusion-instruct}}
}
```

## Contact
For questions or feedback, please open an issue on the Hugging Face repository or the [source GitHub repository](https://github.com/yarenty/trainer).