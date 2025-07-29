import os
from huggingface_hub import HfApi, login
from transformers import AutoModelForCausalLM, AutoTokenizer
from trainer import config

# User must be logged in via huggingface-cli login
# Or set HF_TOKEN as env variable

def main():
    # Set your Hugging Face repo name (e.g., "username/model-name")
    REPO_ID = os.environ.get("HF_REPO_ID") or "yarenty/" + config.FINE_TUNED_MODEL
    MODEL_DIR = os.path.join(config.MODELS_DIR, config.FINE_TUNED_MODEL)
    TOKENIZER_DIR = MODEL_DIR  # adjust if tokenizer is elsewhere
    GGUF_PATH = os.path.join(config.MODELS_DIR, config.GGUF_MODEL)
    
    

    # Login (optional if already logged in)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # # Upload model (transformers format)
    # print(f"Uploading model from {MODEL_DIR} to {REPO_ID}...")
    # model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    # tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    # model.push_to_hub(REPO_ID)
    # tokenizer.push_to_hub(REPO_ID)
    # print(f"Model and tokenizer uploaded to https://huggingface.co/{REPO_ID}")

    # Optionally upload GGUF or other files
    # if os.path.exists(GGUF_PATH):
    #     print(f"Uploading GGUF file {GGUF_PATH} to repo as an asset...")
    #     api = HfApi()
    #     api.upload_file(
    #         path_or_fileobj=GGUF_PATH,
    #         path_in_repo=os.path.basename(GGUF_PATH),
    #         repo_id=REPO_ID,
    #         repo_type="model"
    #     )
    #     print(f"GGUF file uploaded to https://huggingface.co/{REPO_ID}/blob/main/{os.path.basename(GGUF_PATH)}")
    
    gguf_model_dir = os.path.join(config.MODELS_DIR, config.GGUF_MODEL)
    gguf_repo_id = os.environ.get("HF_GGUF_REPO_ID") or f"yarenty/{config.GGUF_MODEL}"
    
    if os.path.exists(gguf_model_dir):
        api = HfApi()
        print(f"Creating repository {gguf_repo_id} (if it doesn't exist)...")
        api.create_repo(repo_id=gguf_repo_id, repo_type="model", exist_ok=True)
        
        print(f"Uploading contents of {gguf_model_dir} to {gguf_repo_id}...")
        api.upload_folder(
            folder_path=gguf_model_dir,
            repo_id=gguf_repo_id,
            repo_type="model",
        )
        print(f"Successfully uploaded GGUF model to https://huggingface.co/{gguf_repo_id}")
      
        
if __name__ == "__main__":
    main() 