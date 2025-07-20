"""
deduplicator.py: Deduplicate Q/A pairs in .jsonl files.

This module provides a class to detect and remove duplicate questions and/or answers from Q/A datasets.

Usage:
    from trainer.qa_data_quality.deduplicator import QADeduplicator
    dedup = QADeduplicator()
    report = dedup.deduplicate_all_files()
    print(report)

"""
import os
import json
import glob
import logging
from typing import List, Dict, Any
from trainer.config import DATA_DIR

class QADeduplicator:
    """
    Deduplicates Q/A pairs in all .jsonl files under qa_data/*/*.jsonl.
    By default, removes duplicate questions (case-insensitive, stripped).
    """
    def __init__(self, data_dir: str = DATA_DIR, dedup_on: str = 'question'):
        """
        Args:
            data_dir: Directory containing Q/A data
            dedup_on: Field to deduplicate on ('question', 'answer', or 'both')
        """
        self.data_dir = data_dir
        self.dedup_on = dedup_on
        self.logger = logging.getLogger(self.__class__.__name__)

    def deduplicate_all_files(self) -> Dict[str, Any]:
        """
        Deduplicate all .jsonl files and return a report.
        """
        pattern = os.path.join(self.data_dir, '*', '*.jsonl')
        files = glob.glob(pattern)
        report = {}
        for file in files:
            file_report = self.deduplicate_file(file)
            report[file] = file_report
        return report

    def deduplicate_file(self, filepath: str) -> Dict[str, Any]:
        """
        Deduplicate Q/A pairs in a single .jsonl file.
        Returns a report of duplicates found and removed.
        """
        seen = set()
        deduped_lines = []
        total = 0
        removed = 0
        issues = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                total += 1
                try:
                    obj = json.loads(line)
                    key = self._dedup_key(obj)
                    if key in seen:
                        removed += 1
                        issues.append(f"Line {i}: Duplicate {self.dedup_on} removed.")
                        continue
                    seen.add(key)
                    deduped_lines.append(json.dumps(obj, ensure_ascii=False) + '\n')
                except Exception as e:
                    issues.append(f"Line {i}: JSON error: {e}")
                    continue
        # Overwrite file with deduplicated lines
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(deduped_lines)
        return {
            'total_lines': total,
            'removed_duplicates': removed,
            'final_lines': len(deduped_lines),
            'issues': issues
        }

    def _dedup_key(self, obj: Dict[str, Any]) -> str:
        """
        Generate a deduplication key for a Q/A pair based on the dedup_on setting.
        """
        if self.dedup_on == 'question':
            return obj.get('question', '').strip().lower()
        elif self.dedup_on == 'answer':
            return obj.get('answer', '').strip().lower()
        elif self.dedup_on == 'both':
            return (obj.get('question', '').strip().lower() + '||' + obj.get('answer', '').strip().lower())
        else:
            return json.dumps(obj, sort_keys=True) 