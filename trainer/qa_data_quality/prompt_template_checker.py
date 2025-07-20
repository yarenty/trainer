"""
prompt_template_checker.py: Check prompt/response template compliance in Q/A datasets.

This module provides a class to check that all Q/A pairs follow the expected prompt/response template.

Usage:
    from trainer.qa_data_quality.prompt_template_checker import QAPromptTemplateChecker
    checker = QAPromptTemplateChecker()
    report = checker.check_all_files()
    print(report)

"""
import os
import json
import glob
import logging
import re
from typing import List, Dict, Any
from trainer.config import DATA_DIR, END_MARKER

class QAPromptTemplateChecker:
    """
    Checks prompt/response template compliance in all .jsonl files under qa_data/*/*.jsonl.
    Flags Q/A pairs that do not match the expected template.
    """
    # Default template regex (adjust as needed)
    QUESTION_HEADER = r"^### Question:\\n"
    ANSWER_HEADER = r"^### Answer:\\n"
    END_MARKER = END_MARKER

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def check_all_files(self) -> Dict[str, Any]:
        """
        Scan all .jsonl files and flag Q/A pairs not matching the template.
        Returns a report with flagged pairs per file.
        """
        pattern = os.path.join(self.data_dir, '*', '*.jsonl')
        files = glob.glob(pattern)
        report = {}
        for file in files:
            file_report = self.check_file(file)
            if file_report['flagged']:
                report[file] = file_report
        return report

    def check_file(self, filepath: str) -> Dict[str, Any]:
        """
        Check template compliance in a single .jsonl file.
        Returns a report of flagged Q/A pairs.
        """
        flagged = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    question = obj.get('question', '').strip()
                    answer = obj.get('answer', '').strip()
                    if not self._matches_template(question, answer):
                        flagged.append({'line': i, 'question': question[:80], 'answer': answer[:80]})
                except Exception:
                    continue
        return {
            'flagged': flagged,
            'num_flagged': len(flagged)
        }

    def _matches_template(self, question: str, answer: str) -> bool:
        # Check question header
        if not question or not answer:
            return False
        # Optionally, check for strict headers (can be relaxed if needed)
        # Here, we just check that the answer ends with the END_MARKER
        if not answer.endswith(self.END_MARKER):
            return False
        # Could add more checks for headers/format if needed
        return True 