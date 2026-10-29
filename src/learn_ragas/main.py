"""
The Learning Hub - Ragas Automated RAG Evaluation Framework CLI.
Demonstrates decoupled metric scoring, hallucination detection, Context Precision@K, and CI/CD Quality Gates.
"""

from __future__ import annotations
import sys
from .roadmap import RAGAS_STAGES, get_completed_count, get_progress_percentage
from .evaluator import (
    SingleTurnSample,
    EvaluationDataset,
    FaithfulnessMetric,
    AnswerRelevanceMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    evaluate
)


def print_banner() -> None:
    print("=" * 76)
    print("   THE LEARNING HUB: RAGAS AUTOMATED RAG EVALUATION ENGINE   ")
    print("=" * 76)


def display_roadmap() -> None:
    print("\n[+] 7-Stage Ragas Pedagogical Mastery Roadmap:")
    for stage in RAGAS_STAGES:
        status = "[DONE]" if stage["completed"] else "[TODO]"
        print(f"  {status} Stage {stage['stage_number']}: {stage['title']}")
        print(f"         Focus: {stage['domain_focus']}")
    print(f"\nProgress: {get_completed_count()}/{len(RAGAS_STAGES)} Stages Complete ({get_progress_percentage()}%)\n")


def build_evaluation_benchmark() -> EvaluationDataset:
    samples = [
        # Sample 1: High quality RAG interaction
        SingleTurnSample(
            user_input="How does PagedAttention eliminate KV cache fragmentation?",
            retrieved_contexts=[
                "PagedAttention allocates physical GPU memory blocks non-contiguously like virtual memory paging.",
                "This design eliminates external memory fragmentation and reduces memory waste to near zero."
            ],
            response="PagedAttention uses virtual memory paging to allocate physical GPU blocks non-contiguously, eliminating external fragmentation.",
            reference="PagedAttention eliminates KV cache fragmentation by using virtual memory pages for non-contiguous GPU block allocation."
        ),
        # Sample 2: Slight Hallucination Test
        SingleTurnSample(
            user_input="What does Reciprocal Rank Fusion do?",
            retrieved_contexts=[
                "Reciprocal Rank Fusion RRF combines rankings from dense and sparse retrievers using formula 1 / (k + rank)."
            ],
            response="Reciprocal Rank Fusion RRF combines rankings using formula 1 / (k + rank). It was invented at Google DeepMind in 2025.",
            reference="Reciprocal Rank Fusion fuses rankings from multiple retrieval systems without needing normalized scores."
        ),
        # Sample 3: Distractor Context (Testing Context Precision)
        SingleTurnSample(
            user_input="What is Direct Preference Optimization DPO?",
            retrieved_contexts=[
                "Kubernetes manages containerized workloads across node clusters.",  # Irrelevant distractor
                "Direct Preference Optimization DPO optimizes LLM policy directly from preference data without a reward model."
            ],
            response="Direct Preference Optimization DPO trains language models directly on preferred chosen vs rejected data without training a separate reward model.",
            reference="DPO optimizes policy directly on human preference pairs without separate reward modeling."
        )
    ]
    return EvaluationDataset(samples)


def demonstrate_evaluation_pipeline(dataset: EvaluationDataset) -> None:
    print("-" * 68)
    print("1. Running Ragas Automated Multi-Metric Evaluation")
    print("-" * 68)

    print(f"  [+] Evaluating benchmark dataset ({len(dataset)} samples)...")
    result = evaluate(dataset)

    print("\n  [Sample-Level Score Breakdown]:")
    for i, (sample, scores) in enumerate(zip(dataset, result.sample_scores)):
        print(f"    Sample {i+1}: '{sample.user_input[:45]}...'")
        for m_name, val in scores.items():
            print(f"      - {m_name:<20}: {val * 100:.1f}%")

    print("\n" + "-" * 68)
    print("2. Global Dataset Scorecard")
    print("-" * 68)
    for m_name, val in result.summary_scores.items():
        bar = "#" * int(val * 20)
        print(f"  {m_name:<22}: {val * 100:5.1f}% [{bar:<20}]")

    print("\n" + "-" * 68)
    print("3. CI/CD Quality Gate Automated Policy Check")
    print("-" * 68)

    thresholds = {
        "faithfulness": 0.70,
        "answer_relevance": 0.50,
        "context_precision": 0.50,
        "context_recall": 0.60,
    }
    passed, violations = result.passes_quality_gate(thresholds)

    if passed:
        print("  [SUCCESS] All production quality gate criteria satisfied!")
        print("            Pull Request approved for deployment.\n")
    else:
        print("  [FAILED] CI/CD Quality Gate Violated:")
        for v in violations:
            print(f"    - Violation: {v}")
        print()


def main() -> None:
    print_banner()
    display_roadmap()
    dataset = build_evaluation_benchmark()
    demonstrate_evaluation_pipeline(dataset)
    print("=" * 76)
    print("   RAGAS EVALUATION FRAMEWORK DEMO COMPLETED SUCCESSFULLY   ")
    print("=" * 76)


if __name__ == "__main__":
    main()
