def generate_markdown_report(
    metrics,
    df_answerable,
    df_unanswerable,
    thresholds
):
    answerable = metrics["answerable_evaluation"]["metrics"]
    unanswerable = metrics["unanswerable_evaluation"]

    report = f"""# RAG Evaluation Report

**Evaluation run:** {metrics["evaluation_run"]}

## Dataset

| Category | Questions |
|---|---:|
| Total | {metrics["dataset"]["total_questions"]} |
| Answerable | {metrics["dataset"]["answerable_questions"]} |
| Unanswerable | {metrics["dataset"]["unanswerable_questions"]} |

---

## Answerable Evaluation

**Framework:** Ragas

| Metric | Score | Threshold | Status |
|---|---:|---:|---|
| Faithfulness | {answerable["faithfulness"]:.4f} | {thresholds["min_faithfulness"]:.2f} | {"PASS" if answerable["faithfulness"] >= thresholds["min_faithfulness"] else "FAIL"} |
| Answer Correctness | {answerable["answer_correctness"]:.4f} | {thresholds["min_answer_correctness"]:.2f} | {"PASS" if answerable["answer_correctness"] >= thresholds["min_answer_correctness"] else "FAIL"} |
| Context Precision | {answerable["context_precision"]:.4f} | {thresholds["min_context_precision"]:.2f} | {"PASS" if answerable["context_precision"] >= thresholds["min_context_precision"] else "FAIL"} |
| Context Recall | {answerable["context_recall"]:.4f} | {thresholds["min_context_recall"]:.2f} | {"PASS" if answerable["context_recall"] >= thresholds["min_context_recall"] else "FAIL"} |

### Failed Answerable Cases

- **{len(df_answerable[df_answerable["faithfulness"] < thresholds["min_faithfulness"]])}** failed faithfulness
- **{len(df_answerable[df_answerable["answer_correctness"] < thresholds["min_answer_correctness"]])}** failed answer correctness
- **{len(df_answerable[df_answerable["context_precision"] < thresholds["min_context_precision"]])}** failed context precision
- **{len(df_answerable[df_answerable["context_recall"] < thresholds["min_context_recall"]])}** failed context recall

---

## Unanswerable Evaluation

**Framework:** Custom no-answer evaluator

| Metric | Score |
|---|---:|
| Safe No-Answer Rate | {unanswerable["metrics"]["safe_no_answer_rate"]:.2%} |
| Unsafe Answer Rate | {unanswerable["metrics"]["unsafe_answer_rate"]:.2%} |

### Outcome Distribution

| Outcome | Count |
|---|---:|
| Clear Abstention | {unanswerable["outcomes"]["clear_abstention"]} |
| Abstained with Grounded Context | {unanswerable["outcomes"]["abstained_with_grounded_context"]} |
| Unsafe Answer | {unanswerable["outcomes"]["unsafe_answer"]} |

---

## Failed Unanswerable Cases

{len(df_unanswerable[df_unanswerable["outcome"] == "unsafe_answer"])} unsafe responses detected.

---

## Evaluation Summary

The evaluation contains **{metrics["dataset"]["total_questions"]} questions**:
**{metrics["dataset"]["answerable_questions"]} answerable** and
**{metrics["dataset"]["unanswerable_questions"]} unanswerable**.

Answerable questions are evaluated using Ragas, while unanswerable
questions are evaluated using a custom abstention and grounding evaluator.
"""

    return report