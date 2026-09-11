# RAG Evaluation Report

**Evaluation run:** 2026-09-11_19-02-50

## Dataset

| Category | Questions |
|---|---:|
| Total | 37 |
| Answerable | 30 |
| Unanswerable | 7 |

---

## Answerable Evaluation

**Framework:** Ragas

| Metric | Score | Threshold | Status |
|---|---:|---:|---|
| Faithfulness | 0.8913 | 0.80 | PASS |
| Answer Correctness | 0.7974 | 0.70 | PASS |
| Context Precision | 0.8495 | 0.80 | PASS |
| Context Recall | 1.0000 | 0.80 | PASS |

### Failed Answerable Cases

- **4** failed faithfulness
- **6** failed answer correctness
- **9** failed context precision
- **0** failed context recall

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
| Clear Abstention | 6 |
| Abstained with Grounded Context | 1 |
| Unsafe Answer | 0 |

---

## Failed Unanswerable Cases

0 unsafe responses detected.

---

## Evaluation Summary

The evaluation contains **37 questions**:
**30 answerable** and
**7 unanswerable**.

Answerable questions are evaluated using Ragas, while unanswerable
questions are evaluated using a custom abstention and grounding evaluator.
