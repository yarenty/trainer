#!/usr/bin/env python3
"""
Test script to verify file writing functionality.
"""

import sys
import json
from pathlib import Path



from trainer.qa_prepare.output_converter import OutputConverter

def test_file_writing():
    """Test that files are written correctly."""
    print("Testing file writing functionality...")
    test_qa_pairs = [
        {"question": "What is Python?", "answer": "Python is a high-level programming language known for its simplicity and readability."},
        {"question": "How do you define a function in Python?", "answer": "You define a function using the 'def' keyword followed by the function name and parameters."}
    ]
    
    # Test output converter
    converter = OutputConverter()
    
    # Test writing to file
    output_file = "test_qa_output.jsonl"
    try:
        converter.write_qa_pairs_to_jsonl(test_qa_pairs, output_file)
        print(f"✓ Successfully wrote {len(test_qa_pairs)} Q&A pairs to {output_file}")
        assert Path(output_file).exists(), "File was not created"
        file_size = Path(output_file).stat().st_size
        print(f"✓ File exists with size: {file_size} bytes")
        assert file_size > 0, "File is empty"
        print("✓ File has content")
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"✓ File contains {len(lines)} lines")
            assert len(lines) == len(test_qa_pairs), f"Expected {len(test_qa_pairs)} lines, got {len(lines)}"
            print("✓ All Q&A pairs were written")
            try:
                first_pair = json.loads(lines[0].strip())
                assert 'question' in first_pair and 'answer' in first_pair, "JSON format is missing required fields"
                print("✓ JSON format is correct")
                print(f"  Sample question: {first_pair['question']}")
                print(f"  Sample answer: {first_pair['answer'][:50]}...")
            except json.JSONDecodeError as e:
                assert False, f"JSON parsing failed: {e}"
    except Exception as e:
        assert False, f"Error writing output: {e}"

def test_merge_functionality():
    """Test merging functionality."""
    print("\nTesting merge functionality...")
    
    # Create two test files
    file1 = "test_file1.jsonl"
    file2 = "test_file2.jsonl"
    merged_file = "test_merged.jsonl"
    converter = OutputConverter()
    
    # Create test data
    qa_pairs1 = [{"question": "Q1", "answer": "A1"}]
    qa_pairs2 = [{"question": "Q2", "answer": "A2"}]
    try:
        # Write individual files
        converter.write_qa_pairs_to_jsonl(qa_pairs1, file1)
        converter.write_qa_pairs_to_jsonl(qa_pairs2, file2)
        
        # Merge files
        converter.merge_jsonl_files([file1, file2], merged_file)
        assert Path(merged_file).exists(), "Merged file was not created"
        with open(merged_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2, f"Merged file has wrong number of lines: {len(lines)}"
            print("✓ Merge functionality works")
        Path(file1).unlink()
        Path(file2).unlink()
        Path(merged_file).unlink()
    except Exception as e:
        assert False, f"Error in merge test: {e}"

def main():
    """Run all tests."""
    print("Testing file writing functionality...\n")
    
    tests = [
        test_file_writing,
        test_merge_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ Test {test.__name__} failed with assertion: {e}")
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All file writing tests passed!")
        print("\nThe issue might be in the Q&A generation or validation logic.")
        print("Try running the modular script with --verbose to see more details:")
        print("python scripts/prepare_data_modular.py --repo-path . --repo-name test --output-dir ./output --verbose")
        return 0
    else:
        print("❌ Some file writing tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 