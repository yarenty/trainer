"""
edge_case_sampler.py: Sample and flag edge cases for human review in Q/A datasets.

This module provides a class to sample edge cases (longest/shortest answers, rarest topic, random) for human review, and is extensible for future LLM-based generation.

Usage:
    from trainer.qa_data_quality.edge_case_sampler import QAEedgeCaseSampler
    sampler = QAEedgeCaseSampler()
    report = sampler.sample_all_files()
    print(report)

"""
import os
import json
import glob
import logging
import random
from typing import List, Dict, Any
from collections import Counter
from trainer.config import DATA_DIR

class QAEedgeCaseSampler:
    """
    Samples edge cases (longest/shortest answers, rarest topic, random) from all .jsonl files under qa_data/*/*.jsonl for human review.
    Extensible for future LLM-based edge case generation.
    """
    def __init__(self, data_dir: str = DATA_DIR, num_random: int = 5):
        self.data_dir = data_dir
        self.num_random = num_random
        self.logger = logging.getLogger(self.__class__.__name__)

    def sample_all_files(self) -> Dict[str, Any]:
        """
        Sample edge cases from all .jsonl files.
        Returns a report with edge cases per file.
        """
        pattern = os.path.join(self.data_dir, '*', '*.jsonl')
        files = glob.glob(pattern)
        report = {}
        for file in files:
            file_report = self.sample_file(file)
            if file_report['edge_cases']:
                report[file] = file_report
        return report

    def sample_file(self, filepath: str) -> Dict[str, Any]:
        """
        Sample edge cases from a single .jsonl file.
        Returns a report of edge cases.
        """
        qa_pairs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    qa_pairs.append({'line': i, 'question': obj.get('question', ''), 'answer': obj.get('answer', ''), 'obj': obj})
                except Exception:
                    continue
        if not qa_pairs:
            return {'edge_cases': []}
        # Longest/shortest answers
        longest = max(qa_pairs, key=lambda x: len(x['answer']))
        shortest = min(qa_pairs, key=lambda x: len(x['answer']))
        # Random sample
        random_sample = random.sample(qa_pairs, min(self.num_random, len(qa_pairs)))
        # Rarest topic (if available)
        topics = [q['obj'].get('topic', None) for q in qa_pairs if 'topic' in q['obj']]
        rarest_topic_case = None
        if topics:
            topic_counts = Counter(topics)
            rarest_topic = min(topic_counts, key=topic_counts.get)
            for q in qa_pairs:
                if q['obj'].get('topic', None) == rarest_topic:
                    rarest_topic_case = q
                    break
        edge_cases = [
            {'type': 'longest', 'line': longest['line'], 'question': longest['question'], 'answer': longest['answer'][:80] + ('...' if len(longest['answer']) > 80 else '')},
            {'type': 'shortest', 'line': shortest['line'], 'question': shortest['question'], 'answer': shortest['answer'][:80] + ('...' if len(shortest['answer']) > 80 else '')},
        ]
        if rarest_topic_case:
            edge_cases.append({'type': 'rarest_topic', 'line': rarest_topic_case['line'], 'question': rarest_topic_case['question'], 'answer': rarest_topic_case['answer'][:80] + ('...' if len(rarest_topic_case['answer']) > 80 else ''), 'topic': rarest_topic})
        for q in random_sample:
            edge_cases.append({'type': 'random', 'line': q['line'], 'question': q['question'], 'answer': q['answer'][:80] + ('...' if len(q['answer']) > 80 else '')})
        return {'edge_cases': edge_cases}

    # Placeholder for future LLM-based edge case generation
    def llm_generate_edge_cases(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        (Future) Use an LLM to generate or flag edge cases for human review.
        """
        raise NotImplementedError("LLM-based edge case generation not yet implemented.") 