"""
run_all_quality_checks.py: Entry point to run all Q/A data quality checks.

This script runs all available data quality, format, and validation checks defined in the qa_data_quality package.
It is designed to be extended as new checks are implemented for each pitfall prevention step.

Usage:
    python -m trainer.qa_data_quality.run_all_quality_checks
"""
import logging
from trainer.qa_data_quality import QAFormatEnforcer, QADeduplicator

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

    # Future steps: Add more checks here as new modules/classes are implemented
    # Example:
    # from trainer.qa_data_quality.balance_checker import QABalanceChecker
    # balance_checker = QABalanceChecker()
    # balance_report = balance_checker.run_all()
    # print("Balance Report:", balance_report)

if __name__ == "__main__":
    main() 