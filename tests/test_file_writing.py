#!/usr/bin/env python3
"""
Test script to verify file writing functionality.
"""

import sys
import json
from pathlib import Path

# # Add src to path
# src_path = Path(__file__).parent / "trainer"
# sys.path.insert(0, str(src_path))

from trainer.qa_prepare.output_converter import OutputConverter

def test_file_writing():
    """Test that files are written correctly."""
    print("Testing file writing functionality...")
    
    # Create test Q&A pairs
    test_qa_pairs = [
        {
            "question": "What is Python?",
            "answer": "Python is a high-level programming language known for its simplicity and readability."
        },
        {
            "question": "How do you define a function in Python?",
            "answer": "You define a function using the 'def' keyword followed by the function name and parameters."
        }
    ]
    
    # Test output converter
    converter = OutputConverter()
    
    # Test writing to file
    output_file = "test_qa_output.jsonl"
    try:
        converter.write_qa_pairs_to_jsonl(test_qa_pairs, output_file)
        print(f"✓ Successfully wrote {len(test_qa_pairs)} Q&A pairs to {output_file}")
        
        # Verify file exists and has content
        if Path(output_file).exists():
            file_size = Path(output_file).stat().st_size
            print(f"✓ File exists with size: {file_size} bytes")
            
            if file_size > 0:
                print("✓ File has content")
                
                # Read and verify content
                with open(output_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    print(f"✓ File contains {len(lines)} lines")
                    
                    if len(lines) == len(test_qa_pairs):
                        print("✓ All Q&A pairs were written")
                        
                        # Parse first line to verify JSON format
                        try:
                            first_pair = json.loads(lines[0].strip())
                            if 'question' in first_pair and 'answer' in first_pair:
                                print("✓ JSON format is correct")
                                print(f"  Sample question: {first_pair['question']}")
                                print(f"  Sample answer: {first_pair['answer'][:50]}...")
                            else:
                                print("✗ JSON format is missing required fields")
                                return False
                        except json.JSONDecodeError as e:
                            print(f"✗ JSON parsing failed: {e}")
                            return False
                    else:
                        print(f"✗ Expected {len(test_qa_pairs)} lines, got {len(lines)}")
                        return False
            else:
                print("✗ File is empty")
                return False
        else:
            print("✗ File was not created")
            return False
            
    except Exception as e:
        print(f"✗ Error writing output: {e}")
        return False
    
    return True

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
        
        # Verify merged file
        if Path(merged_file).exists():
            with open(merged_file, 'r') as f:
                lines = f.readlines()
                if len(lines) == 2:
                    print("✓ Merge functionality works")
                    
                    # Clean up
                    Path(file1).unlink()
                    Path(file2).unlink()
                    Path(merged_file).unlink()
                    return True
                else:
                    print(f"✗ Merged file has wrong number of lines: {len(lines)}")
                    return False
        else:
            print("✗ Merged file was not created")
            return False
            
    except Exception as e:
        print(f"✗ Error in merge test: {e}")
        return False

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
            if test():
                passed += 1
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