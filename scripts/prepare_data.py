import os
import json
import re
from typing import List, Dict, Tuple
import ollama

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
    text = re.sub(r'(\**|__)(.*?)\1', r'\2', text)
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
        # Uncomment the following lines if you have Ollama installed and running
        response = client.chat(model=model_name, messages=[{{'role': 'user', 'content': prompt}}])
        content = response['message']['content']
        qa_pair = json.loads(content)
        return qa_pair
        
        # Placeholder for demonstration without actual LLM call
        # print(f"--- Simulating LLM call for chunk (first 100 chars): {chunk[:100]}...")
        return {
            "question": f"What is the main topic of this section regarding DataFusion?",
            "answer": f"This section discusses {chunk.split('.')[0].strip()} in DataFusion."
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
    Processes documentation for a single repository.
    """
    print(f"\n--- Processing repository: {repo_name} ---")
    
    docs_source_dirs = [
        os.path.join(repo_path, "docs", "source"),
        os.path.join(repo_path, "docs"),
        # Add other potential documentation paths if known for specific repos
    ]

    all_docs_content = ""
    found_docs = False
    for doc_dir in docs_source_dirs:
        if os.path.exists(doc_dir) and os.path.isdir(doc_dir):
            found_docs = True
            for root, _, files in os.walk(doc_dir):
                for file_name in files:
                    if file_name.endswith((".rst", ".md")):
                        file_path = os.path.join(root, file_name)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                all_docs_content += f.read() + "\n\n---\n\n" # Separator between files
                        except Exception as e:
                            print(f"Could not read file {file_path}: {e}")
                            continue
            if all_docs_content.strip(): # If content found in one, no need to check others
                break
    
    if not found_docs or not all_docs_content.strip():
        print(f"No documentation files found or content is empty for {repo_name}. Skipping.")
        return

    cleaned_content = clean_text(all_docs_content)
    if not cleaned_content.strip():
        print(f"Cleaned content is empty for {repo_name}. Skipping.")
        return

    # Define chunking strategies
    chunking_strategies = {
        "heading_based": lambda text: chunk_by_headings(text, min_chars=500),
        "paragraph_based": lambda text: chunk_by_paragraphs(text, min_chars=300),
        "fixed_size_500_50": lambda text: chunk_by_fixed_size(text, chunk_size=500, overlap=50),
    }

    all_qa_pairs = []

    # Initialize Ollama client (uncomment if using Ollama)
    ollama_client = ollama.Client(host='http://localhost:11434') # Adjust host if needed

    for strategy_name, chunk_func in chunking_strategies.items():
        print(f"  Applying chunking strategy: {strategy_name}")
        chunks = chunk_func(cleaned_content)
        print(f"  Generated {len(chunks)} chunks for strategy '{strategy_name}'.")

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            # print(f"  Processing chunk {i+1}/{len(chunks)} from strategy '{strategy_name}'...")
            
            # Call LLM to generate Q&A pair
            qa_pair = generate_qa_with_llm(ollama_client, chunk)
            
            qa_pair["source_repo"] = repo_name
            qa_pair["source_strategy"] = strategy_name
            qa_pair["original_chunk_preview"] = chunk[:200] + "..." if len(chunk) > 200 else chunk
            all_qa_pairs.append(qa_pair)

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
