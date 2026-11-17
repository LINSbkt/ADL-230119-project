# Handbook chatbot evaluation

## Run details

The evaluation was run on 2026-11-16 using `qwen2.5:0.5b` and `nomic-embed-text` through the local Ollama server. The raw answers, retrieval scores, and source pages are saved in `results.json`. The index contained 82 chunks from 30 handbook pages.

The labels below are a manual comparison of each saved answer with `expected_notes` in `questions.json` and the selected handbook. “Correct” means the requested fact is present; “incomplete” means the answer omits a requested detail; “incorrect” means it answers a different question or gives an unsupported conclusion; “ambiguous” means it does not give a reliable handbook-grounded answer. These are qualitative labels for one run, not a statistically meaningful accuracy benchmark.

## Per-question findings

| ID | Label | Finding |
|---|---|---|
| Q1 | Correct | Names both concentrations. |
| Q2 | Correct | Names General Chemistry and Introductory Biology; omits the expected 8-credit total. |
| Q3 | Incorrect | Asked for the major credit total (56); answered with the separate declaration prerequisite. |
| Q4 | Correct | Includes both Capstone courses and the Honor pass requirement; adds related approval and assessment details. |
| Q5 | Correct | Correctly states that at least one of the two foundation courses must be completed before declaring. |
| Q6 | Incorrect | Asked for the IS205 prerequisite (IS105); returned Organic Chemistry learning outcomes instead. |
| Q7 | Incomplete | States that IS101 is offered annually but does not state its prerequisite status. |
| Q8 | Correct | Includes the requested MLO4 ethical-conduct and social-responsibility outcome, with unrelated MLO material. |
| Q9 | Correct | Gives a handbook-consistent interdisciplinary description of the major. |
| Q10 | Incomplete | Describes the minor generally but omits its 24-credit/six-course structure and required course-group breakdown. |
| Q11 | Incomplete | Starts a course list but does not provide the requested complete set of codes, titles, availability, and prerequisites. |
| Q12 | Ambiguous | Correctly notes that the handbook cannot choose for the student, then recommends a concentration without adequate support and ends mid-response. |
| Q13 | Correct | Says tuition is not stated in the handbook. |
| Q14 | Correct | Says the current chair and email are not provided. |

Summary: **8 correct, 2 incorrect, 3 incomplete, and 1 ambiguous** out of 14. Q2 and Q8 contain the core requested fact but omit or add detail; they are counted as correct for the principal fact, not as perfect answers. Q12 is not counted as correct because its final recommendation is unsupported.

## Retrieval observations and limitations

All 14 questions have `used_fallback=false` in `results.json`. The two out-of-document questions (Q13 and Q14) nevertheless receive appropriate refusals from the model. This means those refusals do **not** demonstrate that the current retrieval fallback reliably detects out-of-scope questions. In this implementation, the fallback is based on average similarity score; an unrelated but moderately similar passage can exceed the fixed threshold.

The scores in `results.json` are similarity-derived retrieval scores, not answer-quality scores. For example, Q6 has an average score of 0.6933 while returning the wrong content. Retrieval relevance and answer correctness must therefore be evaluated separately. The Q3/Q5 answers are nearly identical despite different questions, and Q6 suggests that retrieving a relevant course page alone is insufficient if generation does not extract the requested catalogue field.

This is one qualitative run with a small local model and a fixed set of 14 questions. It does not establish general accuracy or prove that another model, prompt, or index configuration would perform better.

## Reproduce the evaluation

From `source_code/`, with the `rag` Python environment active and Ollama running with both models available:

```powershell
python evaluate.py --rebuild
```

The command overwrites `results.json`. Preserve a copy first if you need to retain this run for comparison.
