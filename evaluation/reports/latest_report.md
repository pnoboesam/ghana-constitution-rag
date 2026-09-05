# RAG Evaluation Report

**Evaluation run:** 2026-09-05_22-22-51

## Dataset

| Category | Questions |
|---|---:|
| Total | 5 |
| Answerable | 3 |
| Unanswerable | 2 |

---

## Answerable Evaluation

**Framework:** Ragas

| Metric | Score | Threshold | Status |
|---|---:|---:|---|
| Faithfulness | 0.9074 | 0.80 | PASS |
| Answer Correctness | 0.8846 | 0.70 | PASS |
| Context Precision | 0.9444 | 0.80 | PASS |
| Context Recall | 1.0000 | 0.80 | PASS |

### Failed Answerable Cases

- **0** failed faithfulness
- **0** failed answer correctness
- **0** failed context precision
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
| Clear Abstention | 2 |
| Abstained with Grounded Context | 0 |
| Unsafe Answer | 0 |

---

## Failed Unanswerable Cases

0 unsafe responses detected.

---

## Evaluation Summary

The evaluation contains **5 questions**:
**3 answerable** and
**2 unanswerable**.

Answerable questions are evaluated using Ragas, while unanswerable
questions are evaluated using a custom abstention and grounding evaluator.
