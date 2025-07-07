import os
import json
import re
from typing import List, Dict, Tuple

import ollama # Ensure this is at the top

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing common Markdown/reStructuredText formatting.
    """
    # Remove reStructuredText roles and directives (e.g., :ref:`...`, .. toctree::)
    text = re.sub(r':`.*?`', '', text)
    text = re.sub(r'.. toctree::.*', '', text, flags=re.DOTALL)
    text = re.sub(r'.. image::.*', '', text)
    text = re.sub(r'.. raw:: html.*', '', text, flags=re.DOTALL)
    text = re.sub(r'.. _toc\..*:', '', text)
    text = re.sub(r'.. _.*:', '', text) # General reStructuredText references

    # Remove Markdown/reStructuredText headings (lines starting with #, =, -, ~, etc.)
    text = re.sub(r'^\s*#+\s.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-=~`]+\s*$', '', text, flags=re.MULTILINE)

    # Remove Markdown bold/italic (**, __, *, _)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)

    # Remove Markdown links [text](url) and images ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*\)', '', text)
    text = re.sub(r'\[.*?\]\(.*\)', '', text)

    # Remove code blocks (```...``` or ```python...```)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'``.*?``', '', text) # Inline code

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove multiple newlines and leading/trailing whitespace from lines
    text = os.linesep.join([s.strip() for s in text.splitlines() if s.strip()])
    text = re.sub(r'\n\n+', '\n\n', text) # Reduce multiple newlines to two

    return text.strip()

def chunk_by_headings(text: str, min_chars: int = 200) -> List[str]:
    """
    Chunks text based on Markdown/reStructuredText-like headings.
    Combines smaller chunks to meet a minimum character count.
    """
    # This regex attempts to capture content between headings.
    # It's a simplification and might need refinement for complex docs.
    # For RST, headings are typically underlined, so we look for lines followed by a line of symbols.
    # For MD, headings start with #.
    chunks = re.split(r'\n([#=~-]+[ ]?.+\n[=~-]+\n|\n#+ .+\n)', text)
    
    processed_chunks = []
    current_chunk = ""
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        
        # If it's a heading, append it to the current_chunk if it's not empty, then start a new one
        if re.match(r'^[#=~-]+[ ]?.+\n[=~-]+\n|^#+ .+\n', chunk):
            if current_chunk:
                processed_chunks.append(current_chunk.strip())
            current_chunk = chunk.strip() + "\n"
        else:
            current_chunk += chunk
            if len(current_chunk) >= min_chars:
                processed_chunks.append(current_chunk.strip())
                current_chunk = ""
    
    if current_chunk:
        processed_chunks.append(current_chunk.strip())

    # A second pass to merge very small chunks that might have been split by the regex
    final_chunks = []
    temp_chunk = ""
    for chunk in processed_chunks:
        if not temp_chunk:
            temp_chunk = chunk
        else:
            if len(temp_chunk) + len(chunk) < min_chars * 1.5: # Merge if not too big
                temp_chunk += "\n\n" + chunk
            else:
                final_chunks.append(temp_chunk)
                temp_chunk = chunk
    if temp_chunk:
        final_chunks.append(temp_chunk)

    return [c for c in final_chunks if c]


def chunk_by_paragraphs(text: str, min_chars: int = 100) -> List[str]:
    """
    Chunks text by paragraphs (double newline), merging small paragraphs.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if not current_chunk:
            current_chunk = para
        else:
            if len(current_chunk) + len(para) + 2 <= min_chars * 2: # Allow chunks up to 2x min_chars
                current_chunk += "\n\n" + para
            else:
                chunks.append(current_chunk)
                current_chunk = para
    if current_chunk:
        chunks.append(current_chunk)
    
    return [c for c in chunks if c]

def chunk_by_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Chunks text into fixed-size segments with optional overlap.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            chunks.append(text[start:])
            break
        
        # Try to end on a natural break (e.g., end of sentence/paragraph)
        break_points = [
            text.rfind('\n\n', start, end),
            text.rfind('.', start, end),
            text.rfind('!', start, end),
            text.rfind('?', start, end)
        ]
        break_points = [bp for bp in break_points if bp != -1 and bp > start]
        
        if break_points:
            split_at = max(break_points) + 1
            chunks.append(text[start:split_at].strip())
            start = split_at - overlap
        else:
            chunks.append(text[start:end].strip())
            start = end - overlap
        
        if start < 0: # Ensure start doesn't go negative due to overlap
            start = 0
            
    return [c for c in chunks if c]

def extract_json_from_response(text: str) -> str:
    """
    Extract JSON from a Markdown code block or from the first {...} block in the string.
    """
    # Try to extract JSON from a Markdown code block
    code_block = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', text)
    if code_block:
        return code_block.group(1).strip()
    # Fallback: extract first {...} block
    brace_block = re.search(r'({[\s\S]+})', text)
    if brace_block:
        return brace_block.group(1).strip()
    return text.strip()

def generate_qa_with_llm(client, chunk: str, model_name: str = "llama3.2") -> Dict[str, str]:
    """
    Generates a question and answer pair for a given text chunk using an LLM.
    This function is a placeholder and requires an actual LLM client.
    """
    prompt = f"""Given the following text, generate one concise question and its corresponding answer.
    The question should be directly answerable from the text.
    Format your response as a JSON object with 'question' and 'answer' keys.

    Text:
    ---
    {chunk}
    ---

    JSON:
    """
    
    try:
        # Explicitly ensure prompt is a string
        prompt_for_ollama = str(prompt)

        # Construct messages list
        messages_to_send = [{'role': 'user', 'content': prompt_for_ollama}]

        print(f"DEBUG: Attempting ollama.chat with model='{model_name}' and messages={messages_to_send[0]['content'][:100]}...")

        # The actual Ollama API call
        response = client.chat(model=model_name, messages=messages_to_send)

        print(f"DEBUG: Ollama response type: {type(response)}")
        print(f"DEBUG: Ollama response dir: {dir(response)}")

        # Try to extract the content from the response object
        content = None
        if hasattr(response, "message"):
            print(f"DEBUG: Ollama response.message: {response.message}")
            message = response.message
            if hasattr(message, "content"):
                content = message.content
                print(f"DEBUG: Extracted content from response.message.content: {str(content)[:200]}...")
            else:
                content = str(message)
                print(f"DEBUG: Extracted content from response.message (str): {str(content)[:200]}...")
        elif hasattr(response, "content"):
            content = response.content
            print(f"DEBUG: Extracted content from response.content: {str(content)[:200]}...")
        elif hasattr(response, "model_dump"):
            content_dict = response.model_dump()
            print(f"DEBUG: model_dump: {content_dict}")
            # Try to extract content from dict
            if isinstance(content_dict, dict):
                if "message" in content_dict and isinstance(content_dict["message"], dict):
                    content = content_dict["message"].get("content", None)
                elif "content" in content_dict:
                    content = content_dict["content"]
            print(f"DEBUG: Extracted content from model_dump: {str(content)[:200]}...")
        else:
            print("DEBUG: Unknown response structure from Ollama client. Using str(response)")
            content = str(response)

        # Now parse content as JSON if it's a string
        if isinstance(content, str):
            content = extract_json_from_response(content)
            try:
                qa_pair = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Error: LLM output was not valid JSON. Details: {e}")
                print(f"Problematic content: {content}")
                return {
                    "question": "Error: LLM output was not valid JSON.",
                    "answer": f"Error: LLM output was not valid JSON. Original chunk: {chunk[:200]}..."
                }
        elif isinstance(content, dict):
            qa_pair = content
        else:
            print("Error: Unknown content type from Ollama client.")
            return {
                "question": "Error: Unknown content type from Ollama client.",
                "answer": f"Error: Unknown content type. Original chunk: {chunk[:200]}..."
            }

        return qa_pair

    except TypeError as e:
        print(f"Error: Caught TypeError during Ollama call: {e}")
        print(f"DEBUG: Type of client: {type(client)}")
        print(f"DEBUG: Type of model_name: {type(model_name)}")
        print(f"DEBUG: Type of messages_to_send: {type(messages_to_send)}")
        if messages_to_send and isinstance(messages_to_send[0], dict):
            print(f"DEBUG: First message dict: {messages_to_send[0]}")
        return {
            "question": "Error: Unhashable type dict encountered.",
            "answer": f"Error: Unhashable type dict. Original chunk: {chunk[:200]}..."
        }
    except Exception as e:
        print(f"Error generating QA for chunk: {e}")
        # Fallback or error handling
        return {
            "question": "Error: Could not generate question.",
            "answer": f"Error: Could not generate answer. Original chunk: {chunk[:200]}..."
        }

def process_repository_docs(repo_name: str, repo_path: str, output_base_dir: str):
    """
    Processes documentation for a single repository, skipping irrelevant files and generating Q&A for each doc file.
    """
    print(f"\n--- Processing repository: {repo_name} ---")

    docs_source_dirs = [
        os.path.join(repo_path, "docs", "source"),
        os.path.join(repo_path, "docs"),
        # Add other potential documentation paths if known for specific repos
    ]

    # Filenames to skip (case-insensitive)
    skip_files = {"license", "notice", "contributing", "code_of_conduct"}
    skip_exts = {".txt", ".md", ".rst"}  # Only skip by name, not by extension

    # Define chunking strategies
    chunking_strategies = {
        "heading_based": lambda text: chunk_by_headings(text, min_chars=500),
        "paragraph_based": lambda text: chunk_by_paragraphs(text, min_chars=300),
        "fixed_size_500_50": lambda text: chunk_by_fixed_size(text, chunk_size=500, overlap=50),
    }

    all_qa_pairs = []

    # Initialize Ollama client (uncomment if using Ollama)
    ollama_client = ollama.Client(host='http://localhost:11434') # Adjust host if needed

    found_docs = False
    for doc_dir in docs_source_dirs:
        if os.path.exists(doc_dir) and os.path.isdir(doc_dir):
            found_docs = True
            for root, _, files in os.walk(doc_dir):
                for file_name in files:
                    # Skip irrelevant files by name
                    base_name = os.path.splitext(file_name)[0].lower()
                    if base_name in skip_files:
                        print(f"  Skipping file (by name): {file_name}")
                        continue
                    # Only process .md and .rst files
                    if not (file_name.endswith(".md") or file_name.endswith(".rst")):
                        continue
                    file_path = os.path.join(root, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                    except Exception as e:
                        print(f"Could not read file {file_path}: {e}")
                        continue

                    cleaned_content = clean_text(file_content)
                    if not cleaned_content.strip():
                        print(f"  Cleaned content is empty for {file_name}. Skipping.")
                        continue

                    # Skip if the cleaned content is likely a license or legal text
                    if ("copyright" in cleaned_content.lower() or "license" in cleaned_content.lower()) and len(cleaned_content) < 2000:
                        print(f"  Skipping file (license/legal detected): {file_name}")
                        continue

                    for strategy_name, chunk_func in chunking_strategies.items():
                        print(f"    Applying chunking strategy: {strategy_name} to {file_name}")
                        chunks = chunk_func(cleaned_content)
                        print(f"    Generated {len(chunks)} chunks for strategy '{strategy_name}' in file '{file_name}'.")

                        for i, chunk in enumerate(chunks):
                            if not chunk.strip():
                                continue
                            # Skip chunks that are likely license/legal text
                            chunk_lower = chunk.lower()
                            if ("copyright" in chunk_lower or "license" in chunk_lower) and len(chunk) < 2000:
                                print(f"      Skipping chunk {i+1} in {file_name} (license/legal detected)")
                                continue
                            # Call LLM to generate Q&A pair
                            qa_pair = generate_qa_with_llm(ollama_client, chunk)
                            qa_pair["source_repo"] = repo_name
                            qa_pair["source_strategy"] = strategy_name
                            qa_pair["source_file"] = file_name
                            qa_pair["original_chunk_preview"] = chunk[:200] + "..." if len(chunk) > 200 else chunk
                            all_qa_pairs.append(qa_pair)
            # Do not break; process all doc dirs and files

    if not found_docs:
        print(f"No documentation files found for {repo_name}. Skipping.")
        return
    if not all_qa_pairs:
        print(f"No Q&A pairs generated for {repo_name}. Skipping output.")
        return

    output_file = os.path.join(output_base_dir, f"{repo_name}_qa.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for qa in all_qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')

    print(f"  Generated {len(all_qa_pairs)} Q&A pairs for {repo_name}. Output saved to: {output_file}")


def main():
    base_sources_dir = "/opt/ml/trainer/sources"
    output_data_dir = "/opt/ml/trainer/data"
    
    # Ensure output directory exists
    os.makedirs(output_data_dir, exist_ok=True)

    # Iterate through each subdirectory in base_sources_dir
    for item in os.listdir(base_sources_dir):
        item_path = os.path.join(base_sources_dir, item)
        if os.path.isdir(item_path):
            process_repository_docs(item, item_path, output_data_dir)

    print("\nAll repository documentation processing complete.")

if __name__ == "__main__":
    main()