"""
LLM-based Q&A generation with robust fallback mechanisms.
"""

import re
import logging
from typing import Dict, Optional, Any


class LLM_QA:
    """Handles LLM interactions for Q&A generation with fallback mechanisms."""
    
    def __init__(self, client, model_name: str = "llama3.2"):
        """
        Initialize the LLM Q&A generator.
        
        Args:
            client: Ollama client instance
            model_name: Name of the model to use
        """
        self.client = client
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
    
    def generate_qa_pair(self, chunk: str) -> Dict[str, str]:
        """
        Generate a Q&A pair for a given text chunk.
        
        Args:
            chunk: Text chunk to generate Q&A for
            
        Returns:
            Dictionary with 'question' and 'answer' keys
        """
        try:
            # Try primary generation method
            qa_pair = self._generate_with_llm(chunk)
            if qa_pair and self._is_valid_qa(qa_pair):
                return qa_pair
            
            # Try enhancement if answer is too brief
            if qa_pair and self._needs_enhancement(qa_pair):
                enhanced_pair = self._enhance_answer(qa_pair, chunk)
                if enhanced_pair and self._is_valid_qa(enhanced_pair):
                    return enhanced_pair
            
            # Fallback to simple Q&A generation
            return self._create_fallback_qa(chunk)
            
        except Exception as e:
            self.logger.error(f"Error generating Q&A for chunk: {e}")
            return self._create_fallback_qa(chunk)
    
    def _generate_with_llm(self, chunk: str) -> Optional[Dict[str, str]]:
        """
        Generate Q&A pair using LLM.
        
        Args:
            chunk: Text chunk
            
        Returns:
            Q&A pair dictionary or None if failed
        """
        prompt = f"""Given this code/documentation text, create a comprehensive Q&A pair for fine-tuning a coding assistant.

Question: Ask a practical question that a developer might have about this code/concept. The question should be specific and actionable.

Answer: Provide a detailed, educational response that includes:
1. Clear explanation of the concept/code and its purpose
2. Code examples showing practical usage (use markdown code blocks)
3. Best practices, tips, or important considerations
4. Common pitfalls to avoid (if applicable)
5. Related concepts or alternatives (if relevant)

CRITICAL: Your response must be valid JSON. Follow these rules:
- Use ONLY the exact JSON format shown below
- If you include code examples, use markdown code blocks (```code```)
- Escape any quotes within the answer text with backslashes
- Do NOT use YAML-style formatting (no | characters)
- Do NOT use multi-line strings without proper escaping

Text:
---
{chunk}
---

Respond with ONLY valid JSON in this exact format:
{{
  "question": "Your question here",
  "answer": "Your detailed answer with code examples in markdown blocks. Escape any quotes with \\"
}}
"""
        
        try:
            messages = [{'role': 'user', 'content': str(prompt)}]
            response = self.client.chat(model=self.model_name, messages=messages)
            
            # Extract content from response
            content = self._extract_content_from_response(response)
            
            if isinstance(content, str):
                qa_pair = self._extract_qa_from_response(content)
                if qa_pair:
                    self.logger.debug("Successfully extracted Q&A pair from LLM response")
                    return qa_pair
            
            self.logger.error("Failed to extract Q&A from LLM response")
            return None
            
        except Exception as e:
            self.logger.error(f"Error in LLM generation: {e}")
            return None
    
    def _extract_content_from_response(self, response: Any) -> Optional[str]:
        """
        Extract content from various response formats.
        
        Args:
            response: LLM response object
            
        Returns:
            Extracted content string or None
        """
        if hasattr(response, "message"):
            message = response.message
            if hasattr(message, "content"):
                return message.content
            else:
                return str(message)
        elif hasattr(response, "content"):
            return response.content
        elif hasattr(response, "model_dump"):
            content_dict = response.model_dump()
            if isinstance(content_dict, dict):
                if "message" in content_dict and isinstance(content_dict["message"], dict):
                    return content_dict["message"].get("content")
                elif "content" in content_dict:
                    return content_dict["content"]
        else:
            return str(response)
    
    def _extract_qa_from_response(self, text: str) -> Optional[Dict[str, str]]:
        """
        Extract question and answer directly from LLM response using regex patterns.
        
        Args:
            text: LLM response text
            
        Returns:
            Q&A pair dictionary or None
        """
        self.logger.debug(f"Extracting Q&A from response: {text[:200]}...")
        
        # Try various regex patterns
        patterns = [
            (r'"question":\s*"([^"]*)"', r'"answer":\s*\|(.*?)(?=\s*"|$)', re.DOTALL),
            (r'"question":\s*"([^"]*)"', r'"answer":\s*"([^"]*)"', re.DOTALL),
            (r'question["\s]*:["\s]*([^"\n]+)', r'answer["\s]*:["\s]*\|(.*?)(?=\n\s*["\w]|$)', re.DOTALL),
            (r'question["\s]*:["\s]*([^"\n]+)', r'answer["\s]*:["\s]*([^"\n]+)', re.DOTALL),
        ]
        
        for question_pattern, answer_pattern, flags in patterns:
            try:
                question_match = re.search(question_pattern, text, flags)
                answer_match = re.search(answer_pattern, text, flags)
                
                if question_match and answer_match:
                    question = question_match.group(1).strip()
                    answer = self._clean_answer_text(answer_match.group(1).strip())
                    
                    if question and answer:
                        return {"question": question, "answer": answer}
            except Exception as e:
                self.logger.debug(f"Pattern failed: {e}")
                continue
        
        # Fallback extraction
        return self._fallback_extraction(text)
    
    def _fallback_extraction(self, text: str) -> Optional[Dict[str, str]]:
        """
        Fallback method to extract Q&A from text.
        
        Args:
            text: Response text
            
        Returns:
            Q&A pair dictionary or None
        """
        lines = text.split('\n')
        question = None
        answer_lines = []
        in_answer = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for question
            if not question and ('?' in line or line.lower().startswith(('question', 'how', 'what'))):
                if 'question' in line.lower():
                    question = re.sub(r'^.*?question["\s]*:["\s]*', '', line, flags=re.IGNORECASE).strip()
                else:
                    question = line
                continue
            
            # Collect answer lines
            if question and not in_answer:
                if 'answer' in line.lower() or '|' in line:
                    in_answer = True
                    continue
            
            if in_answer:
                if line.startswith('"') and ('"question":' in line or '"source_' in line):
                    break
                answer_lines.append(line)
        
        if question and answer_lines:
            answer = self._clean_answer_text('\n'.join(answer_lines))
            if answer:
                return {"question": question, "answer": answer}
        
        return None
    
    def _clean_answer_text(self, answer: str) -> str:
        """
        Clean up answer text.
        
        Args:
            answer: Raw answer text
            
        Returns:
            Cleaned answer text
        """
        answer = answer.strip()
        
        # Remove YAML pipe characters
        if answer.startswith('|'):
            answer = answer[1:].strip()
        
        # Remove JSON artifacts
        answer = re.sub(r'^\s*"', '', answer)
        answer = re.sub(r'"\s*$', '', answer)
        
        # Clean up whitespace
        answer = re.sub(r'\n\s*\n', '\n\n', answer)
        
        return answer
    
    def _needs_enhancement(self, qa_pair: Dict[str, str]) -> bool:
        """
        Check if answer needs enhancement.
        
        Args:
            qa_pair: Q&A pair dictionary
            
        Returns:
            True if enhancement is needed
        """
        answer = qa_pair.get('answer', '')
        return (
            len(answer) < 300 or
            answer.count('.') < 3 or
            ('```' not in answer and '`' not in answer) or
            answer.count('\n') < 2
        )
    
    def _enhance_answer(self, qa_pair: Dict[str, str], chunk: str) -> Optional[Dict[str, str]]:
        """
        Enhance a brief answer with more details.
        
        Args:
            qa_pair: Original Q&A pair
            chunk: Original text chunk
            
        Returns:
            Enhanced Q&A pair or None
        """
        enhancement_prompt = f"""The following answer is too brief. Please enhance it with more details, code examples, and educational content.

Question: {qa_pair['question']}
Current Answer: {qa_pair['answer']}

Original Context:
{chunk}

Please provide a comprehensive, enhanced answer that includes:
1. More detailed explanation
2. Code examples or usage patterns (use markdown code blocks)
3. Best practices or tips
4. Common pitfalls or considerations
5. Related concepts if applicable

IMPORTANT: Respond with ONLY the enhanced answer text. Do not include JSON formatting or quotes around the answer.

Enhanced Answer:"""

        try:
            messages = [{'role': 'user', 'content': enhancement_prompt}]
            response = self.client.chat(model=self.model_name, messages=messages)
            
            content = self._extract_content_from_response(response)
            if content:
                enhanced_answer = content.strip()
                if enhanced_answer.startswith("Enhanced Answer:"):
                    enhanced_answer = enhanced_answer[len("Enhanced Answer:"):].strip()
                
                if enhanced_answer.startswith('"') and enhanced_answer.endswith('"'):
                    enhanced_answer = enhanced_answer[1:-1]
                
                if len(enhanced_answer) > len(qa_pair['answer']):
                    return {
                        "question": qa_pair['question'],
                        "answer": enhanced_answer
                    }
            
            return qa_pair
            
        except Exception as e:
            self.logger.error(f"Error enhancing answer: {e}")
            return qa_pair
    
    def _create_fallback_qa(self, chunk: str) -> Dict[str, str]:
        """
        Create a fallback Q&A pair when LLM generation fails.
        
        Args:
            chunk: Text chunk
            
        Returns:
            Fallback Q&A pair
        """
        try:
            lines = chunk.split('\n')
            first_line = lines[0].strip() if lines else ""
            
            # Determine question type based on content
            if "fn " in chunk:
                question = "What does this function do and how is it used?"
            elif "struct " in chunk:
                question = "What is this struct and what are its components?"
            elif "impl " in chunk:
                question = "What does this implementation provide?"
            else:
                question = "What is the purpose of this code?"
            
            answer = f"This code appears to be related to: {first_line[:100]}... "
            answer += "Please refer to the original documentation for complete details and usage examples."
            
            return {"question": question, "answer": answer}
            
        except Exception as e:
            self.logger.error(f"Error creating fallback Q&A: {e}")
            return {
                "question": "Error: Could not generate question.",
                "answer": f"Error: Could not generate answer. Original chunk: {chunk[:200]}..."
            }
    
    def _is_valid_qa(self, qa_pair: Dict[str, str]) -> bool:
        """
        Validate Q&A pair quality.
        
        Args:
            qa_pair: Q&A pair dictionary
            
        Returns:
            True if Q&A pair is valid
        """
        if not isinstance(qa_pair, dict):
            return False
        
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        
        # Check for error messages
        if 'error:' in question.lower() or 'error:' in answer.lower():
            return False
        
        # Check minimum requirements
        if len(question) < 10 or len(answer) < 20:
            return False
        
        return True 