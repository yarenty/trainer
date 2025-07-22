---
license: meta-llama/llama-3-2
base_model: meta-llama/Llama-3.2-8B-Instruct
tags:
- text-generation
- instruction
- datafusion
- rust
- code
---

![transformers](https://img.shields.io/badge/transformers-yes-green)
![Downloads](https://img.shields.io/endpoint?url=https://huggingface.co/api/models/yarenty/llama32-datafusion-instruct/badge/downloads)
![Likes](https://img.shields.io/endpoint?url=https://huggingface.co/api/models/yarenty/llama32-datafusion-instruct/badge/likes)

**Author:** yarenty  
**Model type:** Llama 3.2 (fine-tuned)  
**Task:** Instruction-following, code Q/A, DataFusion expert assistant  
**License:** Apache 2.0  
**Visibility:** Public

---

# Llama 3.2 DataFusion Instruct

This model is a fine-tuned version of **meta-llama/Llama-3.2-8B-Instruct**, specialized for the [Apache Arrow DataFusion](https://arrow.apache.org/datafusion/) ecosystem. It's designed to be a helpful assistant for developers, answering technical questions, generating code, and explaining concepts related to DataFusion, Arrow.rs, Ballista, and the broader Rust data engineering landscape.

**GGUF Version:** For quantized, low-resource deployment, you can find the GGUF version [here](<https://huggingface.co/yarenty/llama32-datafusion-instruct-gguf>).

## Model Description

This model was fine-tuned on a curated dataset of high-quality question-answer pairs and instruction-following examples sourced from the official DataFusion documentation, source code, mailing lists, and community discussions.

- **Model Type:** Instruction-following Large Language Model (LLM)
- **Base Model:** `meta-llama/Llama-3.2-8B-Instruct`
- **Primary Use:** Developer assistant for the DataFusion ecosystem.

## Prompt Template

To get the best results, format your prompts using the following instruction template.

```
### Instruction:
{Your question or instruction here}

### Response:
```

## Example Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "yarenty/llama32-datafusion-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# The model was trained with a specific instruction template.
# For optimal performance, your prompt should follow this structure.
prompt_template = """### Instruction:
How do I register a Parquet file in DataFusion?

### Response:"""

inputs = tokenizer(prompt_template, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id)

# Decode the output, skipping special tokens and the prompt
prompt_length = inputs["input_ids"].shape[1]
print(tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True))
```

## Training Procedure

- **Hardware:** Trained on 1x NVIDIA A100 GPU.
- **Training Script:** Custom script using `transformers.SFTTrainer`.
- **Key Hyperparameters:**
  - Epochs: 3
  - Learning Rate: 2e-5
  - Batch Size: 4
- **Dataset:** A curated dataset of ~5,000 high-quality QA pairs and instructions related to DataFusion. Data was cleaned and deduplicated as per the notes in `pitfalls.md`.

## Intended Use & Limitations

- **Intended Use:** This model is intended for developers and data engineers working with DataFusion. It can be used for code generation, debugging assistance, and learning the library. It can also serve as a strong base for further fine-tuning on more specialized data.
- **Limitations:** The model's knowledge is limited to the data it was trained on. It may produce inaccurate or outdated information for rapidly evolving parts of the library. It is not a substitute for official documentation or expert human review.

## Citation

If you find this model useful in your work, please cite:
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