![transformers](https://img.shields.io/badge/transformers-yes-green)
![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)
![Downloads](https://img.shields.io/endpoint?url=https://huggingface.co/api/models/yarenty/llama32-datafusion-instruct/badge/downloads)
![Likes](https://img.shields.io/endpoint?url=https://huggingface.co/api/models/yarenty/llama32-datafusion-instruct/badge/likes)

# Llama 3.2 DataFusion Instruct

**Author:** yarenty  
**Model type:** Llama 3.2 (fine-tuned)  
**Task:** Instruction-following, code Q/A, DataFusion expert assistant  
**License:** Apache 2.0  
**Visibility:** Public

---

## Model Description

This model is a fine-tuned version of Llama 3.2, trained on high-quality question/answer pairs and instruction-following data from the DataFusion open source ecosystem. It is designed to:
- Answer technical questions about DataFusion, Rust, and related data engineering topics
- Provide code examples, best practices, and explanations
- Support instruction-style prompts for code generation and troubleshooting

## Metrics

| Metric         | Value   | Dataset         |
|----------------|---------|----------------|
| Exact Match    | 82.5%   | datafusion-qa   |
| F1 Score       | 89.1%   | datafusion-qa   |
| Human Rating   | 4.5/5   | Internal eval   |

*Note: These are example values. Replace with your actual evaluation results if available.*

## Training Data
- Q/A pairs generated from DataFusion documentation, code, and real-world usage
- Instruction/input/output format for advanced use cases
- Emphasis on context-rich, multi-sentence answers with code and references

## Intended Use
- As an assistant for developers working with DataFusion and Rust
- For code generation, debugging, and learning
- As a base for further fine-tuning or integration into RAG systems

## Limitations
- May hallucinate or provide outdated information for rapidly evolving libraries
- Not a replacement for official documentation or expert review
- English language only

## Example Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("yarenty/llama32-datafusion-instruct")
tokenizer = AutoTokenizer.from_pretrained("yarenty/llama32-datafusion-instruct")

prompt = "How do I register a Parquet file in DataFusion?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation
If you use this model, please cite:
```
@misc{yarenty_llama32_datafusion_instruct,
  title={Llama 3.2 DataFusion Instruct},
  author={yarenty},
  year={2024},
  howpublished={\url{https://huggingface.co/yarenty/llama32-datafusion-instruct}}
}
```

## Contact
For questions or feedback, open an issue on the Hugging Face repo or contact yarenty. 