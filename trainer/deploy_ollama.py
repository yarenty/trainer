import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import subprocess
from config import MODELS_DIR, DEFAULT_MODEL_NAME, MERGED_MODEL, GGUF_MODEL, FINAL_OLLAMA, FINE_TUNED_MODEL, LOCAL_MODEL_DIR

def main():
    # --- Configuration ---
    # Path to your original base model (e.g., Llama 3.2 from Hugging Face)
    # This should match the model_name used in scripts/train.py
    

    base_model_name = LOCAL_MODEL_DIR #"meta-llama/Llama-2-7b-hf" # Replace with Llama 3.2 when available on HF or use a local path

    # Path where your fine-tuned LoRA adapters were saved by scripts/train.py
    finetuned_adapter_path = os.path.join(MODELS_DIR, FINE_TUNED_MODEL)

    # Directory to save the merged Hugging Face model
    merged_model_output_path = os.path.join(MODELS_DIR, MERGED_MODEL)

    # Directory to save the GGUF file and Modelfile
    gguf_output_dir = os.path.join(MODELS_DIR, GGUF_MODEL)
    
    # !!! IMPORTANT: Set this to the absolute path of your cloned llama.cpp directory !!!
    llama_cpp_path = "/opt/ml/trainer/llama.cpp" # <--- CHANGE THIS LINE

    # Name for your model in Ollama
    ollama_model_name = FINAL_OLLAMA

    # Quantization type for GGUF (e.g., q4_k_m, q5_k_m, q8_0)
    # q4_k_m is a good balance of size and performance
    gguf_quant_type = "q4_k_m"

    # --- Check required files and directories ---
    quantize_bin = os.path.join(llama_cpp_path, "build", "bin", "llama-quantize")
    quantize_exec_path = None
    if os.path.isfile(quantize_bin):
        quantize_exec_path = quantize_bin
    else:
        print(f"Error: quantize executable not found at {quantize_bin}. Please ensure it is built (look for 'llama-quantize').")
        return

    required_paths = [
        (base_model_name, True, "Base model directory"),
        (finetuned_adapter_path, True, "Fine-tuned adapter directory"),
        (llama_cpp_path, True, "llama.cpp directory"),
        (os.path.join(llama_cpp_path, "convert_hf_to_gguf.py"), False, "convert_hf_to_gguf.py script"),
    ]
    for path, is_dir, desc in required_paths:
        if is_dir and not os.path.isdir(path):
            print(f"Error: {desc} not found at {path}. Please ensure it exists.")
            return
        if not is_dir and not os.path.isfile(path):
            print(f"Error: {desc} not found at {path}. Please ensure it exists.")
            return

    # --- Create directories ---
    os.makedirs(merged_model_output_path, exist_ok=True)
    os.makedirs(gguf_output_dir, exist_ok=True)

    # --- 1. Load Base Model and LoRA Adapters ---
    print(f"Loading base model from local directory: {base_model_name} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16, # Use float16 for efficiency
        device_map="auto", # Load model onto available devices
        local_files_only=True, # Ensure only local files are used
    )
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True)

    print(f"Loading LoRA adapters from {finetuned_adapter_path} and merging...")
    model = PeftModel.from_pretrained(base_model, finetuned_adapter_path)

    # --- 2. Merge Adapters ---
    # This merges the LoRA weights into the base model weights
    model = model.merge_and_unload()
    print("LoRA adapters merged into the base model.")

    # --- 3. Save Merged Model ---
    print(f"Saving merged Hugging Face model to {merged_model_output_path}...")
    model.save_pretrained(merged_model_output_path)
    base_tokenizer.save_pretrained(merged_model_output_path)
    print("Merged Hugging Face model saved.")

    # --- 4. GGUF Conversion ---
    # Paths to llama.cpp conversion scripts/executables
    convert_py_path = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    # quantize_exec_path is already set above

    if not os.path.exists(convert_py_path):
        print(f"Error: {convert_py_path} not found. Please ensure llama.cpp is cloned and built correctly.")
        return

    # Define intermediate and final GGUF paths
    intermediate_gguf_path = os.path.join(gguf_output_dir, "model.gguf") # convert.py directly outputs GGUF now
    final_gguf_path = os.path.join(gguf_output_dir, f"{ollama_model_name}.gguf")

    print(f"Converting merged model to GGUF format using {convert_py_path}...")
    convert_command = [
        "python3", # Use python3 explicitly
        convert_py_path,
        "--model", merged_model_output_path,
        "--outfile", intermediate_gguf_path,
        "--outtype", "f16" # Convert to float16 GGUF first
    ]
    print(f"Running command: {' '.join(convert_command)}")
    try:
        subprocess.run(convert_command, check=True, cwd=llama_cpp_path)
        print("Conversion to intermediate GGUF (f16) successful.")
    except subprocess.CalledProcessError as e:
        print(f"Error during GGUF conversion: {e}")
        print(f"Stdout: {e.stdout.decode() if e.stdout else ''}")
        print(f"Stderr: {e.stderr.decode() if e.stderr else ''}")
        return

    print(f"Quantizing GGUF model to {gguf_quant_type} using {quantize_exec_path}...")
    quantize_command = [
        quantize_exec_path,
        intermediate_gguf_path,
        final_gguf_path,
        gguf_quant_type
    ]
    print(f"Running command: {' '.join(quantize_command)}")
    try:
        subprocess.run(quantize_command, check=True, cwd=llama_cpp_path)
        print(f"Quantization to {gguf_quant_type} successful. Final GGUF saved to {final_gguf_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during GGUF quantization: {e}")
        print(f"Stdout: {e.stdout.decode()}")
        print(f"Stderr: {e.stderr.decode()}")
        return
    
    # Clean up intermediate GGUF file
    if os.path.exists(intermediate_gguf_path):
        os.remove(intermediate_gguf_path)
        print(f"Removed intermediate GGUF file: {intermediate_gguf_path}")


    # --- 5. Create Ollama Modelfile ---
    modelfile_content = f"""FROM {final_gguf_path}
PARAMETER stop "### Instruction:"
PARAMETER stop "### Response:"
PARAMETER stop "### End"
"""
    modelfile_path = os.path.join(gguf_output_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    print(f"Modelfile created at {modelfile_path}")

    # --- 6. Ollama Import Instructions ---
    print("\n--- Ollama Import Instructions ---")
    print("To import your fine-tuned model into Ollama, run the following command:")
    print(f"ollama create {ollama_model_name} -f {modelfile_path}")
    print(f"\nAfter import, you can run your model with: ollama run {ollama_model_name}")
    print("Example query: ollama run datafusion-llama3 \"### Instruction:\nHow do I use the filter operation in DataFusion?\n\n### Response:\"")

if __name__ == "__main__":
    main()
