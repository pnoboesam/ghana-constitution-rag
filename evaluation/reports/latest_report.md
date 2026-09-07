# RAG Evaluation Report

**Evaluation run:** 2026-09-07_16-03-13

## Dataset

| Category | Questions |
|---|---:|
| Total | 37 |
| Answerable | 31 |
| Unanswerable | 6 |

---

## Answerable Evaluation

**Framework:** Ragas

| Metric | Score | Threshold | Status |
|---|---:|---:|---|
| Faithfulness | 0.8197 | 0.80 | PASS |
| Answer Correctness | 0.7423 | 0.70 | PASS |
| Context Precision | 0.9613 | 0.80 | PASS |
| Context Recall | 0.9839 | 0.80 | PASS |

### Failed Answerable Cases

- **13** failed faithfulness
- **10** failed answer correctness
- **2** failed context precision
- **1** failed context recall

---

## Unanswerable Evaluation

**Framework:** Custom no-answer evaluator

| Metric | Score |
|---|---:|
| Safe No-Answer Rate | 100.00% |
| Unsafe Answer Rate | 0.00% |

### Outcome Distribution

| Outcome | Count |
|---|---:|
| Clear Abstention | 5 |
| Abstained with Grounded Context | 1 |
| Unsafe Answer | 0 |

---

## Failed Unanswerable Cases

0 unsafe responses detected.

---

## Evaluation Summary

The evaluation contains **37 questions**:
**31 answerable** and
**6 unanswerable**.

Answerable questions are evaluated using Ragas, while unanswerable
questions are evaluated using a custom abstention and grounding evaluator.
