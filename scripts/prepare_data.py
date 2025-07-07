import os
import json
import re
from typing import List, Dict, Tuple
import glob
import concurrent.futures
from concurrent.futures import as_completed
import logging

import ollama # Ensure this is at the top

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

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
    MAX_CHUNKS = 10000
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
        if len(processed_chunks) > MAX_CHUNKS:
            logging.error(f"chunk_by_headings: Too many chunks (> {MAX_CHUNKS}). Aborting.")
            return []
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
        if len(final_chunks) > MAX_CHUNKS:
            logging.error(f"chunk_by_headings: Too many chunks after merge (> {MAX_CHUNKS}). Aborting.")
            return []
    if temp_chunk:
        final_chunks.append(temp_chunk)
    logging.debug(f"chunk_by_headings: Generated {len(final_chunks)} chunks. First chunk preview: {final_chunks[0][:100] if final_chunks else 'EMPTY'}")
    if len(final_chunks) > MAX_CHUNKS:
        logging.error(f"chunk_by_headings: Final chunk count exceeds {MAX_CHUNKS}. Returning empty list.")
        return []
    return [c for c in final_chunks if c]


def chunk_by_paragraphs(text: str, min_chars: int = 100) -> List[str]:
    """
    Chunks text by paragraphs (double newline), merging small paragraphs.
    """
    MAX_CHUNKS = 10000
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
        if len(chunks) > MAX_CHUNKS:
            logging.error(f"chunk_by_paragraphs: Too many chunks (> {MAX_CHUNKS}). Aborting.")
            return []
    if current_chunk:
        chunks.append(current_chunk)
    logging.debug(f"chunk_by_paragraphs: Generated {len(chunks)} chunks. First chunk preview: {chunks[0][:100] if chunks else 'EMPTY'}")
    if len(chunks) > MAX_CHUNKS:
        logging.error(f"chunk_by_paragraphs: Final chunk count exceeds {MAX_CHUNKS}. Returning empty list.")
        return []
    return [c for c in chunks if c]

def chunk_by_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Chunks text into fixed-size segments with optional overlap.
    """
    MAX_CHUNKS = 10000
    if overlap >= chunk_size:
        logging.error(f"chunk_by_fixed_size: overlap ({overlap}) >= chunk_size ({chunk_size}). Aborting.")
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        if end > text_len:
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
            if split_at <= start:
                logging.warning(f"chunk_by_fixed_size: split_at ({split_at}) <= start ({start}). Advancing by chunk_size.")
                split_at = end
            chunks.append(text[start:split_at].strip())
            new_start = split_at - overlap
        else:
            chunks.append(text[start:end].strip())
            new_start = end - overlap
        if new_start <= start:
            logging.error(f"chunk_by_fixed_size: new_start ({new_start}) <= start ({start}). Aborting to prevent infinite loop.")
            break
        start = new_start
        if len(chunks) > MAX_CHUNKS:
            logging.error(f"chunk_by_fixed_size: Too many chunks (> {MAX_CHUNKS}). Aborting.")
            return []
    logging.debug(f"chunk_by_fixed_size: Generated {len(chunks)} chunks. First chunk preview: {chunks[0][:100] if chunks else 'EMPTY'}")
    if len(chunks) > MAX_CHUNKS:
        logging.error(f"chunk_by_fixed_size: Final chunk count exceeds {MAX_CHUNKS}. Returning empty list.")
        return []
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

        logging.debug(f"Attempting ollama.chat with model='{model_name}' and messages={messages_to_send[0]['content'][:100]}...")

        # The actual Ollama API call
        response = client.chat(model=model_name, messages=messages_to_send)

        logging.debug(f"Ollama response type: {type(response)}")
        logging.debug(f"Ollama response dir: {dir(response)}")

        # Try to extract the content from the response object
        content = None
        if hasattr(response, "message"):
            logging.debug(f"Ollama response.message: {response.message}")
            message = response.message
            if hasattr(message, "content"):
                content = message.content
                logging.debug(f"Extracted content from response.message.content: {str(content)[:200]}...")
            else:
                content = str(message)
                logging.debug(f"Extracted content from response.message (str): {str(content)[:200]}...")
        elif hasattr(response, "content"):
            content = response.content
            logging.debug(f"Extracted content from response.content: {str(content)[:200]}...")
        elif hasattr(response, "model_dump"):
            content_dict = response.model_dump()
            logging.debug(f"model_dump: {content_dict}")
            # Try to extract content from dict
            if isinstance(content_dict, dict):
                if "message" in content_dict and isinstance(content_dict["message"], dict):
                    content = content_dict["message"].get("content", None)
                elif "content" in content_dict:
                    content = content_dict["content"]
            logging.debug(f"Extracted content from model_dump: {str(content)[:200]}...")
        else:
            logging.debug("Unknown response structure from Ollama client. Using str(response)")
            content = str(response)

        # Now parse content as JSON if it's a string
        if isinstance(content, str):
            content = extract_json_from_response(content)
            try:
                qa_pair = json.loads(content)
            except json.JSONDecodeError as e:
                logging.error(f"LLM output was not valid JSON. Details: {e}")
                logging.error(f"Problematic content: {content}")
                return {
                    "question": "Error: LLM output was not valid JSON.",
                    "answer": f"Error: LLM output was not valid JSON. Original chunk: {chunk[:200]}..."
                }
        elif isinstance(content, dict):
            qa_pair = content
        else:
            logging.error("Unknown content type from Ollama client.")
            return {
                "question": "Error: Unknown content type from Ollama client.",
                "answer": f"Error: Unknown content type. Original chunk: {chunk[:200]}..."
            }

        return qa_pair

    except TypeError as e:
        logging.error(f"Error: Caught TypeError during Ollama call: {e}")
        logging.error(f"Type of client: {type(client)}")
        logging.error(f"Type of model_name: {type(model_name)}")
        logging.error(f"Type of messages_to_send: {type(messages_to_send)}")
        if messages_to_send and isinstance(messages_to_send[0], dict):
            logging.error(f"First message dict: {messages_to_send[0]}")
        return {
            "question": "Error: Unhashable type dict encountered.",
            "answer": f"Error: Unhashable type dict. Original chunk: {chunk[:200]}..."
        }
    except Exception as e:
        logging.error(f"Error generating QA for chunk: {e}")
        # Fallback or error handling
        return {
            "question": "Error: Could not generate question.",
            "answer": f"Error: Could not generate answer. Original chunk: {chunk[:200]}..."
        }

def process_single_doc_file(file_path, file_name, repo_name, chunking_strategies, clean_text, generate_qa_with_llm, ollama_client, output_file):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
    except Exception as e:
        logging.error(f"Could not read file {file_path}: {e}")
        return
    cleaned_content = clean_text(file_content)
    if not cleaned_content.strip():
        logging.info(f"Cleaned content is empty for {file_name}. Skipping.")
        return
    # Skip if the cleaned content is likely a license or legal text
    if ("copyright" in cleaned_content.lower() or "license" in cleaned_content.lower()) and len(cleaned_content) < 2000:
        logging.info(f"Skipping file (license/legal detected): {file_name}")
        return
    for strategy_name, chunk_func in chunking_strategies.items():
        logging.info(f"Applying chunking strategy: {strategy_name} to {file_name}")
        chunks = chunk_func(cleaned_content)
        logging.debug(f"{strategy_name}: {len(chunks)} chunks for {file_name}")
        if not chunks:
            logging.warning(f"{strategy_name}: No chunks generated for {file_name}. Skipping this strategy.")
            continue
        if len(chunks) > 10000:
            logging.error(f"{strategy_name}: Too many chunks ({len(chunks)}) for {file_name}. Skipping this strategy.")
            continue
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            # Skip chunks that are likely license/legal text
            chunk_lower = chunk.lower()
            if ("copyright" in chunk_lower or "license" in chunk_lower) and len(chunk) < 2000:
                logging.info(f"Skipping chunk {i+1} in {file_name} (license/legal detected)")
                continue
            logging.debug(f"Processing chunk {i+1}/{len(chunks)} (size: {len(chunk)})")
            qa_pair = generate_qa_with_llm(ollama_client, chunk)
            qa_pair["source_repo"] = repo_name
            qa_pair["source_strategy"] = strategy_name
            qa_pair["source_file"] = file_name
            qa_pair["original_chunk_preview"] = chunk[:200] + "..." if len(chunk) > 200 else chunk
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')

def process_single_code_file(file_path, file_name, repo_name, code_chunking_strategy, clean_text, generate_qa_with_llm, ollama_client, output_file):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
    except Exception as e:
        logging.error(f"Could not read code file {file_path}: {e}")
        return
    cleaned_content = clean_text(file_content)
    if not cleaned_content.strip():
        logging.info(f"Cleaned content is empty for code file {file_name}. Skipping.")
        return
    logging.info(f"Chunking code file: {file_name}")
    chunks = code_chunking_strategy(cleaned_content)
    logging.debug(f"code_fixed_size_500_50: {len(chunks)} chunks for {file_name}")
    if not chunks:
        logging.warning(f"code_fixed_size_500_50: No chunks generated for {file_name}. Skipping.")
        return
    if len(chunks) > 10000:
        logging.error(f"code_fixed_size_500_50: Too many chunks ({len(chunks)}) for {file_name}. Skipping.")
        return
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        logging.debug(f"Processing chunk {i+1}/{len(chunks)} (size: {len(chunk)})")
        qa_pair = generate_qa_with_llm(ollama_client, chunk)
        qa_pair["source_repo"] = repo_name
        qa_pair["source_strategy"] = "code_fixed_size_500_50"
        qa_pair["source_file"] = file_name
        qa_pair["original_chunk_preview"] = chunk[:200] + "..." if len(chunk) > 200 else chunk
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')

def process_repository_docs(repo_name: str, repo_path: str, output_base_dir: str):
    """
    Processes documentation and code source files for a single repository, skipping irrelevant files and generating Q&A for each doc/code file.
    Appends each Q&A pair to the output file immediately after generation.
    """
    logging.info(f"\n--- Processing repository: {repo_name} ---")
    docs_source_dirs = [
        os.path.join(repo_path, "docs", "source"),
        os.path.join(repo_path, "docs"),
        # Add other potential documentation paths if known for specific repos
    ]
    code_source_dirs = [repo_path]
    for subdir in ["src", "examples"]:
        subdir_path = os.path.join(repo_path, subdir)
        if os.path.exists(subdir_path) and os.path.isdir(subdir_path) and subdir_path not in code_source_dirs:
            code_source_dirs.append(subdir_path)
    processed_code_files = set()
    skip_files = {"license", "notice", "contributing", "code_of_conduct"}
    skip_exts = {".txt", ".md", ".rst"}  # Only skip by name, not by extension
    chunking_strategies = {
        "heading_based": lambda text: chunk_by_headings(text, min_chars=500),
        "paragraph_based": lambda text: chunk_by_paragraphs(text, min_chars=300),
        "fixed_size_500_50": lambda text: chunk_by_fixed_size(text, chunk_size=500, overlap=50),
    }
    code_chunking_strategy = lambda text: chunk_by_fixed_size(text, chunk_size=500, overlap=50)
    code_file_extensions = {".rs", ".py", ".c", ".cpp", ".h", ".hpp"}
    ollama_client = ollama.Client(host='http://localhost:11434') # Adjust host if needed
    found_docs = False
    output_file = os.path.join(output_base_dir, f"{repo_name}_qa.jsonl")
    # Collect all doc and code file tasks
    future_to_fileinfo = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Docs
        for doc_dir in docs_source_dirs:
            if os.path.exists(doc_dir) and os.path.isdir(doc_dir):
                found_docs = True
                for root, _, files in os.walk(doc_dir):
                    for file_name in files:
                        base_name = os.path.splitext(file_name)[0].lower()
                        if base_name in skip_files:
                            logging.info(f"Skipping file (by name): {file_name}")
                            continue
                        if not (file_name.endswith(".md") or file_name.endswith(".rst")):
                            continue
                        file_path = os.path.join(root, file_name)
                        future = executor.submit(
                            process_single_doc_file,
                            file_path, file_name, repo_name, chunking_strategies,
                            clean_text, generate_qa_with_llm, ollama_client, output_file
                        )
                        future_to_fileinfo[future] = (file_name, 'doc')
        # Code
        for code_dir in code_source_dirs:
            for root, _, files in os.walk(code_dir):
                for file_name in files:
                    ext = os.path.splitext(file_name)[1].lower()
                    if ext not in code_file_extensions:
                        continue
                    file_path = os.path.join(root, file_name)
                    abs_path = os.path.abspath(file_path)
                    if abs_path in processed_code_files:
                        continue
                    processed_code_files.add(abs_path)
                    future = executor.submit(
                        process_single_code_file,
                        file_path, file_name, repo_name, code_chunking_strategy,
                        clean_text, generate_qa_with_llm, ollama_client, output_file
                    )
                    future_to_fileinfo[future] = (file_name, 'code')
        # Process results as they complete
        for future in as_completed(future_to_fileinfo):
            file_name, file_type = future_to_fileinfo[future]
            try:
                future.result(timeout=120)
            except concurrent.futures.TimeoutError:
                logging.warning(f"Timeout: Skipping {file_type} file {file_name} after 2 minutes.")
            except Exception as e:
                logging.error(f"Error processing {file_type} file {file_name}: {e}")
    if not found_docs:
        logging.warning(f"No documentation files found for {repo_name}. Skipping.")
        return
    logging.info(f"Q&A pairs appended to: {output_file}")


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

    logging.info("\nAll repository documentation processing complete.")


if __name__ == "__main__":
    main()