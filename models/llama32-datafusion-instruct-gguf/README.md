# Llama 3.2 DataFusion Instruct (GGUF)

This is a fine-tuned Llama 3.2 model for DataFusion code and Q&A, exported in GGUF format for use with compatible inference engines (e.g., llama.cpp, Ollama).

## Model Details

- **Base model:** Llama 3.2
- **Fine-tuned for:** DataFusion, code Q&A, instruction following
- **Format:** GGUF
- **Stop sequences:**
  - `### Instruction:`
  - `### Response:`
- **Files included:**
  - `llama32_datafusion.gguf` (main model)
  - `Modelfile` (Ollama/llama.cpp config)

## Usage

### With Ollama

```bash
ollama create llama32-datafusion-instruct-gguf -f Modelfile
ollama run llama32-datafusion-instruct-gguf
```

### With llama.cpp

```bash
./main -m llama32_datafusion.gguf --stop "### Instruction:" --stop "### Response:"
```

## Training Data

- Q&A pairs and code from the DataFusion project and related documentation.
- Data cleaning, deduplication, and formatting enforced as per [pitfalls_plan.md](../pitfalls_plan.md).

## License

[Specify your license here, e.g., Apache-2.0, MIT, or custom.]

## Citation

If you use this model, please cite:
```
@misc{llama32-datafusion-instruct-gguf,
  author = {yarenty},
  title = {Llama 3.2 DataFusion Instruct (GGUF)},
  year = {2025},
  howpublished = {Hugging Face},
  url = {https://huggingface.co/yarenty/llama32-datafusion-instruct-gguf}
}
```

## Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/yarenty/trainer) or contact [yarenty@gmail.com](mailto:yarenty@gmail.com). 