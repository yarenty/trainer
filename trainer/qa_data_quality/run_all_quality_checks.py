"""
run_all_quality_checks.py: Entry point to run all Q/A data quality checks.

This script runs all available data quality, format, and validation checks defined in the qa_data_quality package.
It is designed to be extended as new checks are implemented for each pitfall prevention step.

Usage:
    python -m trainer.qa_data_quality.run_all_quality_checks
"""
import logging
from trainer.qa_data_quality import QAFormatEnforcer, QADeduplicator, QABalanceAnalyzer, QAAmbiguityFlagger, QACodeBlockValidator, QAPromptTemplateChecker, QAOutputPostProcessor, QAEedgeCaseSampler, QAQuestionStructureAdjuster

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

    # Step 1b: Adjust question structure to standardized format
    structure_adjuster = QAQuestionStructureAdjuster()
    structure_report = structure_adjuster.adjust_all_files()
    print("\nQuestion Structure Adjustment Report:")
    if structure_report:
        for file, file_report in structure_report.items():
            print(f"\nFile: {file}")
            print(f"  Total lines: {file_report['total_lines']}")
            print(f"  Adjusted lines: {file_report['adjusted_lines']}")
            if file_report['issues']:
                print("  Issues:")
                for issue in file_report['issues']:
                    print(f"    - {issue}")
            else:
                print("  No issues found.")
    else:
        print("  No files processed for structure adjustment.")

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

    # # Step 3: Analyze topic/answer balance
    # analyzer = QABalanceAnalyzer()
    # balance_report = analyzer.analyze_all_files()
    # print("\nBalance Analysis Report:")
    # for topic, stats in balance_report['topic_stats'].items():
    #     print(f"\nTopic: {topic}")
    #     print(f"  Questions: {stats['count']}")
    #     print(f"  Answer length (min/mean/median/max/stdev): {stats['answer_length_min']} / {stats['answer_length_mean']} / {stats['answer_length_median']} / {stats['answer_length_max']} / {stats['answer_length_stdev']}")
    # if balance_report['imbalance_flags']:
    #     print("\nImbalance Flags:")
    #     for flag in balance_report['imbalance_flags']:
    #         print(f"  - {flag}")
    # else:
    #     print("\nNo major imbalances detected.")

    # # Step 4: Flag ambiguous or multi-answer questions
    # flagger = QAAmbiguityFlagger()
    # ambiguity_report = flagger.flag_all_files()
    # print("\nAmbiguity Flagging Report:")
    # if ambiguity_report:
    #     for file, file_report in ambiguity_report.items():
    #         print(f"\nFile: {file}")
    #         print(f"  Flagged questions: {file_report['num_flagged']}")
    #         for flagged in file_report['flagged']:
    #             print(f"    Line {flagged['line']}: {flagged['question']}")
    # else:
    #     print("  No ambiguous or multi-answer questions flagged.")

    # Step 5: Auto-correct code blocks and formatting in answers
    code_validator = QACodeBlockValidator()
    corrections = code_validator.auto_correct_all_files()
    print("\nCode Block Auto-Correction Report:")
    if corrections:
        for file, file_report in corrections.items():
            print(f"\nFile: {file}")
            print(f"  Corrections made: {file_report['num_corrected']}")
            for corr in file_report['corrected']:
                print(f"    Line {corr['line']}: {', '.join(corr['fixes'])}")
                print(f"      Original: {corr['original']}")
                print(f"      Fixed:    {corr['fixed']}")
    else:
        print("  No auto-corrections needed.")

    # Step 5b: Validate code blocks and formatting in answers (after auto-correction)
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

    # # Step 6: Check prompt/response template compliance
    # template_checker = QAPromptTemplateChecker()
    # template_report = template_checker.check_all_files()
    # print("\nPrompt Template Compliance Report:")
    # if template_report:
    #     for file, file_report in template_report.items():
    #         print(f"\nFile: {file}")
    #         print(f"  Flagged pairs: {file_report['num_flagged']}")
    #         for flagged in file_report['flagged']:
    #             print(f"    Line {flagged['line']}: Q: {flagged['question']} | A: {flagged['answer']}")
    # else:
    #     print("  All Q/A pairs match the expected template.")

    # # Step 7: Post-process outputs (trim at end marker, flag verbosity/length issues)
    # postproc = QAOutputPostProcessor()
    # postproc_report = postproc.process_all_files()
    # print("\nOutput Post-Processing Report:")
    # if postproc_report:
    #     for file, file_report in postproc_report.items():
    #         print(f"\nFile: {file}")
    #         print(f"  Flagged outputs: {file_report['num_flagged']}")
    #         for flagged in file_report['flagged']:
    #             print(f"    Line {flagged['line']}: {', '.join(flagged['issues'])}")
    #             print(f"      Preview: {flagged['answer']}")
    # else:
    #     print("  All outputs trimmed and within length limits.")

    # # Step 8: Sample edge cases for human review
    # sampler = QAEedgeCaseSampler()
    # edge_report = sampler.sample_all_files()
    # print("\nEdge Case Sampling Report:")
    # if edge_report:
    #     for file, file_report in edge_report.items():
    #         print(f"\nFile: {file}")
    #         for case in file_report['edge_cases']:
    #             print(f"  [{case['type']}] Line {case['line']}: Q: {case['question'][:60]} | A: {case['answer']}")
    #             if 'topic' in case:
    #                 print(f"    Topic: {case['topic']}")
    # else:
    #     print("  No edge cases sampled.")

    # Future steps: Add more checks here as new modules/classes are implemented
    # Example:
    # from trainer.qa_data_quality.balance_checker import QABalanceChecker
    # balance_checker = QABalanceChecker()
    # balance_report = balance_checker.run_all()
    # print("Balance Report:", balance_report)

if __name__ == "__main__":
    main() 