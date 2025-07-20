"""
run_all_quality_checks.py: Entry point to run all Q/A data quality checks.

This script runs all available data quality, format, and validation checks defined in the qa_data_quality package.
It is designed to be extended as new checks are implemented for each pitfall prevention step.

Usage:
    python -m trainer.qa_data_quality.run_all_quality_checks
"""
import logging
from trainer.qa_data_quality import QAFormatEnforcer, QADeduplicator, QABalanceAnalyzer, QAAmbiguityFlagger, QACodeBlockValidator

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    print("Running Q/A Data Quality Checks...\n")

    # Step 1: Enforce Q/A format and end marker
    enforcer = QAFormatEnforcer()
    report = enforcer.enforce_all_files()
    print("Format Enforcement Report:")
    for file, file_report in report.items():
        print(f"\nFile: {file}")
        print(f"  Total lines: {file_report['total_lines']}")
        print(f"  Fixed lines: {file_report['fixed_lines']}")
        if file_report['issues']:
            print("  Issues:")
            for issue in file_report['issues']:
                print(f"    - {issue}")
        else:
            print("  No issues found.")

    # Step 2: Deduplicate Q/A pairs
    deduplicator = QADeduplicator()
    dedup_report = deduplicator.deduplicate_all_files()
    print("\nDeduplication Report:")
    for file, file_report in dedup_report.items():
        print(f"\nFile: {file}")
        print(f"  Total lines: {file_report['total_lines']}")
        print(f"  Removed duplicates: {file_report['removed_duplicates']}")
        print(f"  Final lines: {file_report['final_lines']}")
        if file_report['issues']:
            print("  Issues:")
            for issue in file_report['issues']:
                print(f"    - {issue}")
        else:
            print("  No issues found.")

    # Step 3: Analyze topic/answer balance
    analyzer = QABalanceAnalyzer()
    balance_report = analyzer.analyze_all_files()
    print("\nBalance Analysis Report:")
    for topic, stats in balance_report['topic_stats'].items():
        print(f"\nTopic: {topic}")
        print(f"  Questions: {stats['count']}")
        print(f"  Answer length (min/mean/median/max/stdev): {stats['answer_length_min']} / {stats['answer_length_mean']} / {stats['answer_length_median']} / {stats['answer_length_max']} / {stats['answer_length_stdev']}")
    if balance_report['imbalance_flags']:
        print("\nImbalance Flags:")
        for flag in balance_report['imbalance_flags']:
            print(f"  - {flag}")
    else:
        print("\nNo major imbalances detected.")

    # Step 4: Flag ambiguous or multi-answer questions
    flagger = QAAmbiguityFlagger()
    ambiguity_report = flagger.flag_all_files()
    print("\nAmbiguity Flagging Report:")
    if ambiguity_report:
        for file, file_report in ambiguity_report.items():
            print(f"\nFile: {file}")
            print(f"  Flagged questions: {file_report['num_flagged']}")
            for flagged in file_report['flagged']:
                print(f"    Line {flagged['line']}: {flagged['question']}")
    else:
        print("  No ambiguous or multi-answer questions flagged.")

    # Step 5: Validate code blocks and formatting in answers
    code_validator = QACodeBlockValidator()
    code_report = code_validator.validate_all_files()
    print("\nCode Block Validation Report:")
    if code_report:
        for file, file_report in code_report.items():
            print(f"\nFile: {file}")
            print(f"  Flagged answers: {file_report['num_flagged']}")
            for flagged in file_report['flagged']:
                print(f"    Line {flagged['line']}: {', '.join(flagged['issues'])}")
                print(f"      Preview: {flagged['answer']}")
    else:
        print("  No code block or formatting issues flagged.")

    # Future steps: Add more checks here as new modules/classes are implemented
    # Example:
    # from trainer.qa_data_quality.balance_checker import QABalanceChecker
    # balance_checker = QABalanceChecker()
    # balance_report = balance_checker.run_all()
    # print("Balance Report:", balance_report)

if __name__ == "__main__":
    main() 