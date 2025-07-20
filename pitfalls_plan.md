# Common Pitfalls in LLM Fine-Tuning: Plan & Prevention

This document outlines the most common pitfalls encountered when fine-tuning large language models (LLMs) for code, Q/A, and instruction-following tasks, along with strategies to prevent them.

---

## 1. Training Data Issues
- **No clear stopping signal:**
  - Problem: Model doesn't know when to stop generating.
  - Prevention: Add explicit end-of-answer marker (e.g., `### End`).
- **Inconsistent formatting:**
  - Problem: Model learns unpredictable stopping points.
  - Prevention: Use strict templates for all Q/A pairs.
- **Too verbose or too short answers:**
  - Problem: Model mimics verbosity or brevity.
  - Prevention: Clean and trim answers to a consistent, reasonable length.
- **Data leakage:**
  - Problem: Test/validation data appears in training, inflating metrics.
  - Prevention: Strictly separate train/val/test sets.
- **Duplicate or near-duplicate entries:**
  - Problem: Overfitting and regurgitation.
  - Prevention: Deduplicate data before training.
- **Unbalanced data:**
  - Problem: Model is biased toward over-represented topics.
  - Prevention: Balance dataset across topics and answer types.
- **Ambiguous or multi-answer questions:**
  - Problem: Model gives hedged or inconsistent answers.
  - Prevention: Remove or clarify ambiguous questions.
- **Poorly cleaned code blocks:**
  - Problem: Model generates invalid code.
  - Prevention: Validate and clean all code in answers.
- **Mixed languages or domains:**
  - Problem: Model blends languages or domains inappropriately.
  - Prevention: Separate or clearly mark different languages/domains.

## 2. Prompt/Template Issues
- **No explicit stop sequence:**
  - Problem: Model keeps generating.
  - Prevention: Set stop sequence in training and inference.
- **Prompt too open-ended:**
  - Problem: Model rambles or is verbose.
  - Prevention: Use prompts with clear instructions and expected format.
- **Prompt/response mismatch:**
  - Problem: Unpredictable model behavior.
  - Prevention: Match inference prompt to training format.
- **Prompt injection risk:**
  - Problem: Model leaks or repeats instructions/code from prompts.
  - Prevention: Clean and review prompt data for injection risks.

## 3. Hyperparameter and Training Issues
- **Too many epochs/too high learning rate:**
  - Problem: Overfitting and memorization.
  - Prevention: Use validation, early stopping, and lower epochs/lr.
- **Batch size too small:**
  - Problem: Unstable training, poor generalization.
  - Prevention: Use reasonable batch size for your hardware.
- **No early stopping or validation monitoring:**
  - Problem: Overfitting goes undetected.
  - Prevention: Monitor validation loss and use early stopping.
- **Not using mixed precision or gradient clipping:**
  - Problem: Instability or slow training.
  - Prevention: Use mixed precision and gradient clipping if possible.

## 4. Data Quality and Preprocessing
- **HTML, escape characters, or artifacts:**
  - Problem: Model outputs artifacts.
  - Prevention: Clean data of HTML, escapes, and artifacts.
- **Inconsistent use of special tokens:**
  - Problem: Model misuses or ignores tokens.
  - Prevention: Use special tokens consistently.
- **Unescaped or mismatched quotes/brackets:**
  - Problem: Model generates broken code/markdown.
  - Prevention: Validate and fix all such issues in data.

## 5. Evaluation and Feedback Issues
- **No real validation set:**
  - Problem: Can't detect overfitting or generalization issues.
  - Prevention: Always use a held-out validation set.
- **No human-in-the-loop evaluation:**
  - Problem: Automated metrics miss subtle quality issues.
  - Prevention: Include human review in evaluation.
- **Ignoring edge cases:**
  - Problem: Model fails on rare or tricky questions.
  - Prevention: Test on edge cases and rare queries.

## 6. Deployment/Inference Issues
- **No max token limit at inference:**
  - Problem: Runaway outputs.
  - Prevention: Set max tokens and stop sequence at inference.
- **No output post-processing:**
  - Problem: Model rambles or never stops.
  - Prevention: Trim/stop at end marker, clean output.
- **Ignoring user context:**
  - Problem: Irrelevant or generic answers.
  - Prevention: Include enough context in prompts.

---

## Summary Table

| Pitfall                        | Prevention/Best Practice                        |
|------------------------------- |------------------------------------------------|
| No stop marker                 | Add explicit end-of-answer marker               |
| Inconsistent formatting        | Use strict templates and enforce with scripts   |
| Data leakage                   | Separate train/val/test, check for overlap      |
| Duplicates                     | Deduplicate before training                     |
| Unbalanced data                | Sample or weight to balance topics              |
| Prompt/response mismatch       | Match inference prompt to training format       |
| Overfitting                    | Use validation, early stopping, lower epochs    |
| No max token limit             | Set max tokens and stop sequence at inference   |
| Poor code formatting           | Clean and validate code blocks                  |
| No human eval                  | Include human review in evaluation              |

---

# Prevention Plan: Step-by-Step Guide to Avoiding Fine-Tuning Pitfalls

This chapter provides a practical, step-by-step guide to preventing the most common pitfalls in LLM fine-tuning for code/Q&A/instruction tasks. For each step, we note what is already implemented in this project, what can be automated, and suggest tools or LLM-based solutions. The goal is to ensure your model is both information-rich and fully functional.

## 1. Standardize Data Format and Mark Stopping Points
- **What’s implemented:**
  - All Q/A pairs use a strict template (question, answer, end marker).
  - End-of-answer marker (`### End`) is enforced via post-processing scripts.
- **Automation:**
  - Use `apply_end_marker` and `clean_outputs.py` to ensure every answer ends with the marker.
- **Tools/LLM suggestions:**
  - Use regex or LLM-based scripts to check for missing or inconsistent markers.

## 2. Clean, Deduplicate, and Balance Data
- **What’s implemented:**
  - Deduplication and filtering scripts (see `post_processing.py`).
- **Automation:**
  - Add scripts to check for duplicate questions/answers and remove them.
  - Use sampling or weighting to balance topics if needed.
- **Tools/LLM suggestions:**
  - Use LLMs to flag ambiguous or multi-answer questions for review.

## 3. Validate and Structure Code Blocks
- **What’s implemented:**
  - Manual review of code blocks in answers.
- **Automation:**
  - Add scripts to check for unclosed code blocks, mismatched quotes/brackets, or broken markdown.
- **Tools/LLM suggestions:**
  - Use LLMs to validate code snippets or even auto-correct formatting.

## 4. Separate and Document Data Splits
- **What’s implemented:**
  - Manual separation of train/val/test sets.
- **Automation:**
  - Add scripts to check for overlap between splits.
- **Tools/LLM suggestions:**
  - Use LLMs to generate synthetic validation/test questions if needed.

## 5. Set Up Prompt and Inference Templates
- **What’s implemented:**
  - Modelfile includes `PARAMETER stop "### End"` for Ollama.
  - Prompt templates match training format.
- **Automation:**
  - Add tests to ensure inference prompts match training templates.
- **Tools/LLM suggestions:**
  - Use LLMs to simulate user queries and check for prompt/response mismatches.

## 6. Monitor Training and Use Validation
- **What’s implemented:**
  - Manual monitoring of training/validation loss.
- **Automation:**
  - Add early stopping and validation loss tracking in training scripts.
- **Tools/LLM suggestions:**
  - Use dashboards or alerting for overfitting detection.

## 7. Limit Output and Post-Process at Inference
- **What’s implemented:**
  - Max token limits and stop sequences set in inference scripts and Modelfile.
- **Automation:**
  - Add post-processing to trim output at the end marker.
- **Tools/LLM suggestions:**
  - Use LLMs to review and rate outputs for verbosity or relevance.

## 8. Human-in-the-Loop and Edge Case Testing
- **What’s implemented:**
  - Manual review of model outputs and edge cases.
- **Automation:**
  - Add scripts to sample and review edge cases regularly.
- **Tools/LLM suggestions:**
  - Use LLMs to generate or flag edge cases for human review.

---

## Key Principle: Information-Rich, Well-Structured Q/A
- **Long, information-rich answers are valuable** as long as they are:
  - Well-structured (clear sections, code blocks, explanations)
  - Marked with a clear end-of-answer signal
  - Not excessively verbose or repetitive
- **Main goal:** A fully functioning model that provides complete, relevant, and concise answers—never cut off, never rambling.

**Use this prevention plan as a checklist and workflow to ensure your fine-tuning process is robust, automated, and produces high-quality, production-ready models.** 

---

# Fine-Tuning Prevention Plan: Implementation Checklist

Use this checklist to track the implementation of each prevention step. Update the status as you progress:

## 1. Standardize Data Format and Mark Stopping Points
- [x] Script to enforce Q/A template and end marker in all data files
- [x] Script/test to flag missing or inconsistent end markers
- [x] Integrate end marker check into data pipeline

## 2. Clean, Deduplicate, and Balance Data
- [x] Deduplication script for questions and answers
- [x] Script to analyze and report topic/answer balance
- [x] Tool to flag ambiguous or multi-answer questions (LLM or heuristic)
- [x] Automate removal of duplicates before training

## 3. Validate and Structure Code Blocks
- [x] Script to check for unclosed code blocks, mismatched quotes/brackets, broken markdown
- [x] Code snippet validator (syntax check for code blocks)
- [ ] (Optional) LLM-based auto-correction for code formatting issues (planned)

## 4. Separate and Document Data Splits
- [ ] Script to check for overlap between train/val/test splits
- [ ] Automated reporting of split statistics (size, overlap, etc.)
- [ ] (Optional) Tool to generate synthetic validation/test questions using LLM

## 5. Set Up Prompt and Inference Templates
- [x] Tests to ensure inference prompts match training template
- [x] Script to check for prompt/response format mismatches
- [ ] (Optional) Simulate user queries and check output format with LLM

## 6. Monitor Training and Use Validation
- [ ] Integrate early stopping and validation loss tracking into training scripts
- [ ] Automated alerts or dashboard for overfitting detection
- [ ] Script regular export of training/validation metrics

## 7. Limit Output and Post-Process at Inference
- [x] Ensure inference scripts enforce max token limits and stop sequences
- [x] Add post-processing to trim output at the end marker
- [x] Script review of outputs for verbosity, relevance, and completeness (LLM-based review planned)

## 8. Human-in-the-Loop and Edge Case Testing
- [x] Script regular sampling and review of edge cases from the data
- [x] Tool to flag or generate edge cases for human review (LLM-based generation planned)
- [ ] Automate periodic human review workflow (e.g., sample N outputs per epoch for review)

---

_Note: LLM-based review and generation features are planned for future extension and are scaffolded in the code._ 

Next steps you might consider:
Add LLM-based review/generation to any of the modules (e.g., for edge case generation, answer quality review, etc.)
Integrate these checks into your CI/CD or data pipeline for automated enforcement
Document the workflow for your team or open source users
Add unit tests for each module to ensure reliability
If you want to proceed with LLM-based review, need help with documentation, or want to automate this as part of a larger workflow, just let me know your priority!
