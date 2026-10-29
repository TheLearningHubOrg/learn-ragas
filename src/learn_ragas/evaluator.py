"""
Production-grade Ragas Automated Evaluation Engine for RAG Systems.

Implements the canonical decoupled RAG Triad + Retrieval metrics:
1. Faithfulness: Verifies generated claims against retrieved context (hallucination detection).
2. Answer Relevance: Evaluates semantic alignment with user question.
3. Context Precision: Evaluates whether relevant chunks are ranked at the top (Precision@K).
4. Context Recall: Measures coverage of ground truth reference sentences in retrieved context.
Includes CI/CD Quality Gate assertions and automated scorecard generation.
"""

from __future__ import annotations
import math
import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


class SingleTurnSample:
    """Represents a single query evaluation sample in a RAG benchmark."""

    def __init__(
        self,
        user_input: str,
        retrieved_contexts: List[str],
        response: str,
        reference: Optional[str] = None,
    ):
        self.user_input = user_input
        self.retrieved_contexts = retrieved_contexts
        self.response = response
        self.reference = reference

    def __repr__(self) -> str:
        return (
            f"SingleTurnSample(input={self.user_input[:30]!r}, "
            f"contexts={len(self.retrieved_contexts)}, resp_len={len(self.response)})"
        )


class EvaluationDataset:
    """Container holding benchmark evaluation samples."""

    def __init__(self, samples: Sequence[SingleTurnSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


# ---------------------------------------------------------------------------
# Metric Formulations
# ---------------------------------------------------------------------------

class BaseMetric:
    """Abstract interface for Ragas evaluation metrics."""
    name: str = "base_metric"

    def score(self, sample: SingleTurnSample) -> float:
        raise NotImplementedError


class FaithfulnessMetric(BaseMetric):
    """
    Measures factual consistency:
    Faithfulness = |Claims grounded in context| / |Total claims in response|
    """
    name: str = "faithfulness"

    @staticmethod
    def _extract_claims(text: str) -> List[str]:
        # Split into sentences or clauses
        clauses = re.split(r"[.\n;]", text)
        return [c.strip() for c in clauses if len(c.strip()) > 5]

    def score(self, sample: SingleTurnSample) -> float:
        claims = self._extract_claims(sample.response)
        if not claims:
            return 1.0

        full_context = " ".join(sample.retrieved_contexts).lower()
        grounded_claims = 0

        for claim in claims:
            tokens = set(re.findall(r"\b\w+\b", claim.lower()))
            # Remove common stop words
            content_tokens = {t for t in tokens if len(t) > 3}
            if not content_tokens:
                grounded_claims += 1
                continue

            # Check if majority of claim content tokens are present in retrieved contexts
            match_count = sum(1 for t in content_tokens if t in full_context)
            if match_count / len(content_tokens) >= 0.5:
                grounded_claims += 1

        return round(grounded_claims / len(claims), 4)


class AnswerRelevanceMetric(BaseMetric):
    """
    Evaluates semantic alignment between response and user input:
    Computes lexical / semantic similarity, penalizing empty or evasive answers.
    """
    name: str = "answer_relevance"

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return set(re.findall(r"\b\w{3,}\b", text.lower()))

    def score(self, sample: SingleTurnSample) -> float:
        q_tokens = self._tokenize(sample.user_input)
        r_tokens = self._tokenize(sample.response)

        if not q_tokens or not r_tokens:
            return 0.0

        overlap = q_tokens & r_tokens
        # Jaccard index combined with length penalty
        relevance = len(overlap) / len(q_tokens | r_tokens)
        scaled_score = min(1.0, relevance * 3.0)  # calibrated scale
        return round(scaled_score, 4)


class ContextPrecisionMetric(BaseMetric):
    """
    Evaluates ranking quality of retrieved contexts:
    Context Precision@K = sum(Precision@k * v_k) / Total Relevant Chunks
    Where v_k in {0, 1} indicates relevance of chunk at rank k.
    """
    name: str = "context_precision"

    def score(self, sample: SingleTurnSample) -> float:
        contexts = sample.retrieved_contexts
        if not contexts:
            return 0.0

        # Ground reference or query used to judge chunk relevance
        ref_text = (sample.reference or sample.user_input).lower()
        ref_tokens = set(re.findall(r"\b\w{3,}\b", ref_text))

        relevant_flags: List[int] = []
        for ctx in contexts:
            c_tokens = set(re.findall(r"\b\w{3,}\b", ctx.lower()))
            is_relevant = 1 if len(ref_tokens & c_tokens) >= 2 else 0
            relevant_flags.append(is_relevant)

        total_relevant = sum(relevant_flags)
        if total_relevant == 0:
            return 0.0

        precisions_sum = 0.0
        running_rel = 0
        for k, rel in enumerate(relevant_flags, start=1):
            if rel:
                running_rel += 1
                precisions_sum += (running_rel / k)

        return round(precisions_sum / total_relevant, 4)


class ContextRecallMetric(BaseMetric):
    """
    Measures retrieval completeness:
    Context Recall = |Reference facts present in contexts| / |Total facts in reference|
    """
    name: str = "context_recall"

    def score(self, sample: SingleTurnSample) -> float:
        if not sample.reference:
            return 1.0

        ref_sentences = [s.strip() for s in sample.reference.split(".") if len(s.strip()) > 5]
        if not ref_sentences:
            return 1.0

        full_context = " ".join(sample.retrieved_contexts).lower()
        found = 0

        for sent in ref_sentences:
            tokens = {t for t in re.findall(r"\b\w+\b", sent.lower()) if len(t) > 3}
            if not tokens:
                found += 1
                continue
            matched = sum(1 for t in tokens if t in full_context)
            if matched / len(tokens) >= 0.5:
                found += 1

        return round(found / len(ref_sentences), 4)


# ---------------------------------------------------------------------------
# Evaluator Runner & Quality Gate
# ---------------------------------------------------------------------------

class EvaluationResult:
    """Encapsulates evaluation results per sample and global dataset averages."""

    def __init__(self, sample_scores: List[Dict[str, float]], summary_scores: Dict[str, float]):
        self.sample_scores = sample_scores
        self.summary_scores = summary_scores

    def passes_quality_gate(self, thresholds: Optional[Dict[str, float]] = None) -> Tuple[bool, List[str]]:
        """Verifies if benchmark scores satisfy CI/CD quality gate thresholds."""
        default_thresholds = {
            "faithfulness": 0.80,
            "answer_relevance": 0.70,
            "context_precision": 0.70,
            "context_recall": 0.75,
        }
        active_thresholds = thresholds or default_thresholds
        failures = []

        for metric, min_val in active_thresholds.items():
            actual = self.summary_scores.get(metric, 0.0)
            if actual < min_val:
                failures.append(f"{metric}: actual {actual:.4f} < required {min_val:.4f}")

        return (len(failures) == 0, failures)

    def __repr__(self) -> str:
        return f"EvaluationResult(metrics={self.summary_scores})"


class RagasEvaluator:
    """Orchestrates multi-metric automated evaluation over benchmark datasets."""

    def __init__(self, metrics: Optional[List[BaseMetric]] = None):
        self.metrics = metrics or [
            FaithfulnessMetric(),
            AnswerRelevanceMetric(),
            ContextPrecisionMetric(),
            ContextRecallMetric(),
        ]

    def evaluate_dataset(self, dataset: EvaluationDataset) -> EvaluationResult:
        sample_results: List[Dict[str, float]] = []
        metric_sums: Dict[str, float] = {m.name: 0.0 for m in self.metrics}

        for sample in dataset:
            row: Dict[str, float] = {}
            for m in self.metrics:
                val = m.score(sample)
                row[m.name] = val
                metric_sums[m.name] += val
            sample_results.append(row)

        n = max(1, len(dataset))
        summary_scores = {k: round(v / n, 4) for k, v in metric_sums.items()}

        return EvaluationResult(sample_scores=sample_results, summary_scores=summary_scores)


def evaluate(
    dataset: EvaluationDataset,
    metrics: Optional[List[BaseMetric]] = None,
) -> EvaluationResult:
    """Convenience functional interface matching standard ragas.evaluate() API."""
    evaluator = RagasEvaluator(metrics=metrics)
    return evaluator.evaluate_dataset(dataset)
