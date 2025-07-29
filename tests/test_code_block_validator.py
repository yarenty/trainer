"""
Test suite for QACodeBlockValidator class, focusing on code block and bracket validation/correction.

This test suite covers:
1. Basic bracket matching
2. Complex nested brackets
3. Mixed bracket types
4. Edge cases and corner cases
5. Real-world code examples
6. Error handling

Each test case includes:
- Description of what is being tested
- Example input
- Expected output
- Explanation of why this is the correct behavior
"""

import unittest
import json
import tempfile
import os
from trainer.qa_data_quality.code_block_validator import QACodeBlockValidator

class TestQACodeBlockValidator(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory and validator instance for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = QACodeBlockValidator(data_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up temporary files after testing."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def create_test_file(self, content: str) -> str:
        """
        Helper method to create a test JSONL file.
        
        Args:
            content: The answer text to include in the test file
            
        Returns:
            Path to the created test file
        """
        os.makedirs(os.path.join(self.temp_dir, "test"), exist_ok=True)
        filepath = os.path.join(self.temp_dir, "test", "test.jsonl")
        with open(filepath, "w") as f:
            f.write(json.dumps({"answer": content}) + "\n")
        return filepath

    def test_basic_bracket_matching(self):
        """
        Test simple cases of matching and mismatched brackets.
        These are the foundational cases that must work correctly.
        """
        test_cases = [
            # Simple missing closing bracket
            {
                "input": "function test( {",
                "expected": "function test( {\n})",
                "description": "Missing single closing parenthesis"
            },
            # Simple missing opening bracket
            {
                "input": "function test )",
                "expected": "function test ",
                "description": "Missing single opening parenthesis"
            },
            # Multiple missing brackets of same type
            {
                "input": "console.log(('test')",
                "expected": "console.log(('test'))",
                "description": "Missing one closing parenthesis with nested brackets"
            }
        ]
        
        for case in test_cases:
            with self.subTest(description=case["description"]):
                filepath = self.create_test_file(case["input"])
                self.validator.auto_correct_file(filepath)
                
                with open(filepath) as f:
                    result = json.loads(f.readline())["answer"]
                self.assertEqual(result, case["expected"])

    def test_nested_brackets(self):
        """
        Test complex nested bracket scenarios.
        These cases verify that the validator correctly handles nested structures
        while maintaining proper bracket hierarchy.
        """
        test_cases = [
            {
                "input": "if (test)) { console.log('hi'); }",
                "expected": "if (test) { console.log('hi'); }",
                "description": "Extra closing parenthesis requiring nested opening"
            },
            {
                "input": "function (test { return x}}))",
                "expected": "function (test { return x})",
                "description": "Missing bracket in nested structure"
            },
            {
                "input": "array[idx[0)]",
                "expected": "array[idx[0]]",
                "description": "Mixed bracket types with nesting"
            }
        ]
        
        for case in test_cases:
            with self.subTest(description=case["description"]):
                filepath = self.create_test_file(case["input"])
                self.validator.auto_correct_file(filepath)
                
                with open(filepath) as f:
                    result = json.loads(f.readline())["answer"]
                self.assertEqual(result, case["expected"])

    def test_mixed_bracket_types(self):
        """
        Test scenarios with multiple types of brackets.
        These cases verify that the validator correctly handles different bracket
        types independently while maintaining their relationships.
        """
        test_cases = [
            {
                "input": "fn test[string] { println!('Hello')})",
                "expected": "fn test[string] { println!('Hello')}",
                "description": "Mixed brackets with extra closing parenthesis"
            },
            {
                "input": "let x = {[1, 2, 3]}};",
                "expected": "let x = {[1, 2, 3]};",
                "description": "Nested mixed brackets with extra closing brace"
            },
            {
                "input": "if (condition] { test();",
                "expected": "if (condition) { test();}",
                "description": "Mismatched bracket types and missing closing brace"
            }
        ]
        
        for case in test_cases:
            with self.subTest(description=case["description"]):
                filepath = self.create_test_file(case["input"])
                self.validator.auto_correct_file(filepath)
                
                with open(filepath) as f:
                    result = json.loads(f.readline())["answer"]
                self.assertEqual(result, case["expected"])

    def test_edge_cases(self):
        """
        Test edge cases and unusual scenarios.
        These cases verify that the validator handles extreme or unusual situations
        gracefully without breaking or producing invalid results.
        """
        test_cases = [
            {
                "input": "",
                "expected": "",
                "description": "Empty string"
            },
            {
                "input": "(((((",
                "expected": "((((()))))",
                "description": "Only opening brackets"
            },
            {
                "input": ")))))",
                "expected": "",
                "description": "Only closing brackets"
            },
            {
                "input": "func([{test}])",
                "expected": "func([{test}])",
                "description": "Already perfectly matched brackets"
            }
        ]
        
        for case in test_cases:
            with self.subTest(description=case["description"]):
                filepath = self.create_test_file(case["input"])
                self.validator.auto_correct_file(filepath)
                
                with open(filepath) as f:
                    result = json.loads(f.readline())["answer"]
                self.assertEqual(result, case["expected"])

    def test_real_world_code(self):
        """
        Test realistic code snippets.
        These cases verify that the validator works correctly with actual code
        patterns that might be encountered in practice.
        """
        test_cases = [
            {
                "input": """
                fn process_data(data: Vec<String>) -> Result<(), Error> {
                    let result = data.iter()
                        .map(|x| x.parse::<i32>))
                        .collect::<Vec<_>>();
                    Ok()
                """,
                "expected": """
                fn process_data(data: Vec<String>) -> Result<(), Error> {
                    let result = data.iter()
                        .map(|x| x.parse::<i32>)
                        .collect::<Vec<_>>();
                    Ok()
                    }""",
                "description": "Rust code with various bracket types and nested structures"
            },
            {
                "input": """
                impl MyStruct {
                    pub fn new() -> Self {
                        Self { data: Vec::new() }
                """,
                "expected": """
                impl MyStruct {
                    pub fn new() -> Self {
                        Self { data: Vec::new() }
}
                    }
                """,
                "description": "Rust implementation block with missing closing braces"
            }
        ]
        
        for case in test_cases:
            with self.subTest(description=case["description"]):
                filepath = self.create_test_file(case["input"])
                self.validator.auto_correct_file(filepath)
                
                with open(filepath) as f:
                    result = json.loads(f.readline())["answer"]
                self.assertEqual(result.strip(), case["expected"].strip())

    def test_quotes_and_backticks(self):
        """
        Test quote and backtick handling.
        """
        test_cases = [
            {
                "input": 'a "b',
                "expected": 'a "b"',
                "description": "Missing closing double quote"
            },
            {
                "input": 'a "b c d ',
                "expected": 'a "b" c d',
                "description": "Missing closing double quote"
            },
            {
                "input": "a 'b",
                "expected": "a 'b'",
                "description": "Missing closing single quote"
            },
            {
                "input": "a `b",
                "expected": "a `b`",
                "description": "Missing closing backtick"
            },
            
            {
                "input": "a `b c d",
                "expected": "a `b` c d",
                "description": "Missing closing backtick"
            },
            {
                "input": 'a "b" \'c\' `d`',
                "expected": 'a "b" \'c\' `d`',
                "description": "All matched"
            }
        ]

        for case in test_cases:
            with self.subTest(description=case["description"]):
                filepath = self.create_test_file(case["input"])
                self.validator.auto_correct_file(filepath)
                with open(filepath) as f:
                    result = json.loads(f.readline())["answer"]
                self.assertEqual(result, case["expected"])

    def test_error_handling(self):
        """
        Test error handling and invalid inputs.
        These cases verify that the validator handles invalid or malformed
        inputs gracefully without crashing.
        """
        test_cases = [
            {
                "input": None,
                "expected": "",
                "description": "None input"
            },
            {
                "input": "function test(] {",
                "expected": "function test() {}",
                "description": "Mismatched bracket types"
            },
            {
                "input": "let x = [1, 2, 3}",
                "expected": "let x = [1, 2, 3]",
                "description": "Wrong closing bracket type"
            }
        ]
        
        for case in test_cases:
            with self.subTest(description=case["description"]):
                if case["input"] is not None:  # Skip None input for file creation
                    filepath = self.create_test_file(case["input"])
                    self.validator.auto_correct_file(filepath)
                    
                    with open(filepath) as f:
                        result = json.loads(f.readline())["answer"]
                    self.assertEqual(result, case["expected"])

    def test_block_closing_brackets(self):
        """
        Test the fix_block_closing_brackets function for code block scenarios.
        """
        from trainer.qa_data_quality.code_block_validator import fix_block_closing_brackets
        test_cases = [
            {
                "input": "fn foo() {\n    let x = bar(\n        1,\n        2\n\n    let y = 3;\n}",
                "expected": "fn foo() {\n    let x = bar(\n        1,\n        2\n    )\n    let y = 3;\n}",
                "description": "Closing bracket before empty line"
            },
            {
                "input": "fn foo() {\n    let x = bar(\n        1,\n        2\n    let y = 3;\n}",
                "expected": "fn foo() {\n    let x = bar(\n        1,\n        2\n    )\n    let y = 3;\n}",
                "description": "Closing bracket before same-indentation line"
            },
            {
                "input": "fn foo() {\n    let x = bar(\n        1,\n        2",
                "expected": "fn foo() {\n    let x = bar(\n        1,\n        2\n    )",
                "description": "Closing bracket at the end if no suitable line"
            },
            {
                "input": "fn foo() {\n    let x = bar(\n        1,\n        2\n    let y = baz[\n        3,\n        4\n    let z = 5;\n}",
                "expected": "fn foo() {\n    let x = bar(\n        1,\n        2\n    )\n    let y = baz[\n        3,\n        4\n    ]\n    let z = 5;\n}",
                "description": "Multiple unclosed brackets in one block"
            }
        ]
        for case in test_cases:
            with self.subTest(description=case["description"]):
                result = fix_block_closing_brackets(case["input"])
                self.assertEqual(result, case["expected"])

if __name__ == '__main__':
    unittest.main() 