from trainer.config import DATA_DIR
import os
import json
from glob import glob

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


def main():
    # Recursively find all .jsonl files in output/ subdirectories
    for dirpath, _, _ in os.walk(DATA_DIR):
        for jsonl_file in glob(os.path.join(dirpath, '*.jsonl')):
            process_jsonl_file(jsonl_file)

if __name__ == "__main__":
    main() 