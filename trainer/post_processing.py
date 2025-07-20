import os
import json
from glob import glob
from trainer import config
from trainer.config import DATA_DIR


TARGET_PHRASE = "Please refer to the original documentation for complete details and usage examples."


def process_jsonl_file(filepath):
    lines_to_keep = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                answer = obj.get("answer", "")
                if TARGET_PHRASE not in answer:
                    lines_to_keep.append(line)
            except json.JSONDecodeError:
                # Keep malformed lines just in case
                lines_to_keep.append(line)
    # Overwrite the file with filtered lines
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines_to_keep)

def apply_end_marker(filepath):
    """
    Go through a .jsonl file and append END_MARKER to each answer if not present.
    """
    end_marker = config.END_MARKER
    updated_lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if 'answer' in obj and obj['answer']:
                    if not obj['answer'].rstrip().endswith(end_marker):
                        obj['answer'] = obj['answer'].rstrip() + f"\n{end_marker}"
                updated_lines.append(json.dumps(obj, ensure_ascii=False) + '\n')
            except json.JSONDecodeError:
                # Keep malformed lines just in case
                updated_lines.append(line)
    # Overwrite the file with updated lines
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)


def main():
    # Recursively find all .jsonl files in output/ subdirectories
    for dirpath, _, _ in os.walk(DATA_DIR):
        for jsonl_file in glob(os.path.join(dirpath, '*.jsonl')):
            process_jsonl_file(jsonl_file)
            apply_end_marker(jsonl_file)

if __name__ == "__main__":
    main() 