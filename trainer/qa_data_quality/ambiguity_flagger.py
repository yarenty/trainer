"""
ambiguity_flagger.py: Flag ambiguous or multi-answer questions in Q/A datasets.

This module provides a class to flag questions that are ambiguous, vague, or likely to have multiple answers.

Usage:
    from trainer.qa_data_quality.ambiguity_flagger import QAAmbiguityFlagger
    flagger = QAAmbiguityFlagger()
    report = flagger.flag_all_files()
    print(report)

"""
import os
import json
import glob
import logging
import re
from typing import List, Dict, Any
from trainer.config import DATA_DIR

class QAAmbiguityFlagger:
    """
    Flags ambiguous or multi-answer questions in all .jsonl files under qa_data/*/*.jsonl.
    """
    # Simple heuristics for ambiguity
    AMBIGUOUS_PATTERNS = [
        r'\bor\b',
        r'could be',
        r'multiple answers',
        r'possible answers',
        r'varies',
        r'it depends',
        r'not clear',
        r'uncertain',
        r'various ways',
        r'(?i)ambiguous',
    ]
    MIN_QUESTION_LENGTH = 10

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def flag_all_files(self) -> Dict[str, Any]:
        """
        Scan all .jsonl files and flag ambiguous/multi-answer questions.
        Returns a report with flagged questions per file.
        """
        pattern = os.path.join(self.data_dir, '*', '*.jsonl')
        files = glob.glob(pattern)
        report = {}
        for file in files:
            file_report = self.flag_file(file)
            if file_report['flagged']:
                report[file] = file_report
        return report

    def flag_file(self, filepath: str) -> Dict[str, Any]:
        """
        Flag ambiguous/multi-answer questions in a single .jsonl file.
        Returns a report of flagged questions.
        """
        flagged = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    question = obj.get('question', '').strip()
                    if self._is_ambiguous(question):
                        flagged.append({'line': i, 'question': question})
                except Exception:
                    continue
        return {
            'flagged': flagged,
            'num_flagged': len(flagged)
        }

    def _is_ambiguous(self, question: str) -> bool:
        if not question or len(question) < self.MIN_QUESTION_LENGTH:
            return True
        for pattern in self.AMBIGUOUS_PATTERNS:
            if re.search(pattern, question, re.IGNORECASE):
                return True
        return False 