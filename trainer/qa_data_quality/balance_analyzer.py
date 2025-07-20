"""
balance_analyzer.py: Analyze and report topic/answer balance in Q/A datasets.

This module provides a class to analyze the distribution of questions/answers by topic and answer length statistics.

Usage:
    from trainer.qa_data_quality.balance_analyzer import QABalanceAnalyzer
    analyzer = QABalanceAnalyzer()
    report = analyzer.analyze_all_files()
    print(report)

"""
import os
import json
import glob
import logging
from typing import List, Dict, Any
from collections import Counter, defaultdict
import statistics
from trainer.config import DATA_DIR

class QABalanceAnalyzer:
    """
    Analyzes topic distribution and answer length statistics in all .jsonl files under qa_data/*/*.jsonl.
    """
    def __init__(self, data_dir: str = DATA_DIR, topic_field: str = None):
        """
        Args:
            data_dir: Directory containing Q/A data
            topic_field: Optional field in JSON to use as topic (otherwise, infer from file name)
        """
        self.data_dir = data_dir
        self.topic_field = topic_field
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_all_files(self) -> Dict[str, Any]:
        """
        Analyze all .jsonl files and return a report.
        """
        pattern = os.path.join(self.data_dir, '*', '*.jsonl')
        files = glob.glob(pattern)
        topic_counts = Counter()
        answer_lengths = defaultdict(list)
        file_reports = {}
        for file in files:
            file_report = self.analyze_file(file)
            file_reports[file] = file_report
            topic = self._infer_topic(file)
            topic_counts[topic] += file_report['num_questions']
            answer_lengths[topic].extend(file_report['answer_lengths'])
        # Aggregate stats
        topic_stats = {topic: {'count': count} for topic, count in topic_counts.items()}
        for topic, lengths in answer_lengths.items():
            if lengths:
                topic_stats[topic]['answer_length_min'] = min(lengths)
                topic_stats[topic]['answer_length_max'] = max(lengths)
                topic_stats[topic]['answer_length_mean'] = round(statistics.mean(lengths), 2)
                topic_stats[topic]['answer_length_median'] = round(statistics.median(lengths), 2)
                if len(lengths) > 1:
                    topic_stats[topic]['answer_length_stdev'] = round(statistics.stdev(lengths), 2)
                else:
                    topic_stats[topic]['answer_length_stdev'] = 0.0
            else:
                topic_stats[topic]['answer_length_min'] = 0
                topic_stats[topic]['answer_length_max'] = 0
                topic_stats[topic]['answer_length_mean'] = 0
                topic_stats[topic]['answer_length_median'] = 0
                topic_stats[topic]['answer_length_stdev'] = 0.0
        # Flag imbalances
        total = sum(topic_counts.values())
        imbalance_flags = []
        for topic, stats in topic_stats.items():
            frac = stats['count'] / total if total else 0
            if frac < 0.05:
                imbalance_flags.append(f"Topic '{topic}' is under-represented ({stats['count']} Qs, {frac:.1%} of total)")
            if frac > 0.5:
                imbalance_flags.append(f"Topic '{topic}' is over-represented ({stats['count']} Qs, {frac:.1%} of total)")
        return {
            'topic_stats': topic_stats,
            'imbalance_flags': imbalance_flags,
            'file_reports': file_reports
        }

    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """
        Analyze a single .jsonl file for number of questions and answer length stats.
        """
        num_questions = 0
        answer_lengths = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    answer = obj.get('answer', '')
                    if answer:
                        num_questions += 1
                        answer_lengths.append(len(answer.strip()))
                except Exception:
                    continue
        return {
            'num_questions': num_questions,
            'answer_lengths': answer_lengths
        }

    def _infer_topic(self, filepath: str) -> str:
        """
        Infer topic from file name or use topic_field if available.
        """
        if self.topic_field:
            # Not implemented: could extract topic from JSON if present
            return self.topic_field
        # Use parent directory as topic (e.g., qa_data/datafusion/code_qa.jsonl -> datafusion)
        return os.path.basename(os.path.dirname(filepath)) 