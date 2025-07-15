#!/usr/bin/env python3
"""
Test script that works without Ollama to verify basic functionality.
"""

import sys
import json
from pathlib import Path

from trainer.qa_prepare.text_cleaner import TextCleaner
from trainer.qa_prepare.chunker import Chunker
from trainer.qa_prepare.output_converter import OutputConverter
from trainer.qa_prepare.llm_qa import LLM_QA

class MockOllamaClient:
    """Mock Ollama client for testing."""
    def chat(self, model, messages):
        # Return a mock response
        class MockResponse:
            def __init__(self):
                self.message = MockMessage()
            
            class MockMessage:
                def __init__(self):
                    self.content = '''
{
  "question": "What is this code about?",
  "answer": "This code demonstrates a basic function that processes data and returns a result. It includes error handling and follows good programming practices."
}
'''
        
        return MockResponse()

def test_full_pipeline():
    """Test the full pipeline without real LLM calls."""
    print("Testing full pipeline with mock LLM...")
    
    # Create test content
    test_content = """
# Sample Python Code

This is a sample Python file for testing.

## Functions

def process_data(input_data):
    \"\"\"
    Process the input data and return a result.
    
    Args:
        input_data: The data to process
        
    Returns:
        Processed result
    \"\"\"
    try:
        result = input_data * 2
        return result
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

## Usage

To use this function:

```python
data = [1, 2, 3, 4, 5]
result = process_data(data)
print(result)
```
"""
    
    # Initialize components
    cleaner = TextCleaner()
    chunker = Chunker()
    converter = OutputConverter()
    
    # Mock LLM client
    mock_client = MockOllamaClient()
    
    # Import LLM_QA and patch it to use mock client
    llm_qa = LLM_QA(mock_client, "test-model")
    
    print("✓ Components initialized")
    
    # Process the content
    print("\nProcessing content...")
    
    # Clean text
    cleaned_content = cleaner.clean_text(test_content)
    print(f"✓ Text cleaned: {len(cleaned_content)} characters")
    
    # Chunk text
    chunks = chunker.chunk_by_headings(cleaned_content, min_chars=50)
    print(f"✓ Text chunked: {len(chunks)} chunks")
    
    if len(chunks) == 0:
        print("❌ No chunks created")
        return False
    
    # Generate Q&A pairs
    qa_pairs = []
    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)}...")
        try:
            qa_pair = llm_qa.generate_qa_pair(chunk)
            if qa_pair and converter.validate_qa_pair(qa_pair):
                # Add metadata
                formatted_pair = converter.format_qa_pair_for_output(
                    qa_pair, "test_file.py", "test-repo"
                )
                qa_pairs.append(formatted_pair)
                print(f"    ✓ Generated Q&A pair")
            else:
                print(f"    ✗ Failed to generate valid Q&A pair")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    print(f"\n✓ Generated {len(qa_pairs)} Q&A pairs")
    
    if len(qa_pairs) == 0:
        print("❌ No Q&A pairs generated")
        return False
    
    # Write output
    output_file = "test_output.jsonl"
    try:
        converter.write_qa_pairs_to_jsonl(qa_pairs, output_file)
        print(f"✓ Wrote output to {output_file}")
        
        # Verify output
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                lines = f.readlines()
                print(f"✓ Output file contains {len(lines)} lines")
                
                # Show first Q&A pair
                if lines:
                    first_pair = json.loads(lines[0])
                    print(f"\nSample Q&A pair:")
                    print(f"  Question: {first_pair.get('question', 'N/A')}")
                    print(f"  Answer: {first_pair.get('answer', 'N/A')[:100]}...")
                    print(f"  Source: {first_pair.get('source_file', 'N/A')}")
                    print(f"  Repo: {first_pair.get('source_repo', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error writing output: {e}")
        return False

def main():
    """Run the test."""
    print("Testing modular pipeline without Ollama...\n")
    
    try:
        success = test_full_pipeline()
        
        if success:
            print("\n🎉 Test passed! The modular structure is working correctly.")
            print("\nTo test with real Ollama:")
            print("1. Make sure Ollama is running")
            print("2. Run: python scripts/prepare_data_modular.py --repo-path . --repo-name test --output-dir ./output --verbose")
            return 0
        else:
            print("\n❌ Test failed. Check the implementation.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 