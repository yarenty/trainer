"""
format_enforcer.py: Enforce Q/A template and end marker in all Q/A data files.

This module provides utilities to check and enforce that all Q/A pairs in .jsonl files under qa_data/*/*.jsonl
conform to the required template and include the configured end marker.

Usage:
    from trainer.qa_data_quality.format_enforcer import QAFormatEnforcer
    enforcer = QAFormatEnforcer()
    report = enforcer.enforce_all_files()
    print(report)

"""
import os
import json
import glob
import logging
from typing import List, Dict, Any
from trainer.config import DATA_DIR, END_MARKER

class QAFormatEnforcer:
    """
    Enforces Q/A template and end marker in all .jsonl files under qa_data/*/*.jsonl.
    """
    def __init__(self, data_dir: str = DATA_DIR, end_marker: str = END_MARKER):
        self.data_dir = data_dir
        self.end_marker = end_marker
        self.logger = logging.getLogger(self.__class__.__name__)

    def enforce_all_files(self) -> Dict[str, Any]:
        """
        Process all .jsonl files and enforce Q/A format and end marker.
        Returns a report dictionary with file stats and issues found/fixed.
        """
        pattern = os.path.join(self.data_dir, '*', '*.jsonl')
        files = glob.glob(pattern)
        report = {}
        for file in files:
            file_report = self.enforce_file(file)
            report[file] = file_report
        return report

    def enforce_file(self, filepath: str) -> Dict[str, Any]:
        """
        Enforce Q/A format and end marker in a single .jsonl file.
        Returns a report of issues found and fixed.
        """
        issues = []
        fixed_lines = []
        total = 0
        fixed = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                total += 1
                try:
                    obj = json.loads(line)
                    question = obj.get('question', '').strip()
                    answer = obj.get('answer', '').strip()
                    # Check for missing fields
                    if not question or not answer:
                        issues.append(f"Line {i}: Missing question or answer.")
                        continue
                    # Enforce end marker
                    if not answer.endswith(self.end_marker):
                        answer = answer.rstrip() + '\n' + self.end_marker
                        obj['answer'] = answer
                        fixed += 1
                        issues.append(f"Line {i}: Added end marker.")
                    # Optionally: enforce template (could add more checks here)
                    fixed_lines.append(json.dumps(obj, ensure_ascii=False) + '\n')
                except Exception as e:
                    issues.append(f"Line {i}: JSON error: {e}")
                    continue
        # Overwrite file with fixed lines
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        return {
            'total_lines': total,
            'fixed_lines': fixed,
            'issues': issues
        } 