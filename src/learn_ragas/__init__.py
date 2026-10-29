"""
The Learning Hub: learn-ragas
Comprehensive guide to automated evaluation of Retrieval-Augmented Generation:
Faithfulness, Answer Relevance, Context Precision, Context Recall, and CI/CD Quality Gates.
"""

from .roadmap import RAGAS_STAGES, get_completed_count, get_progress_percentage
from .evaluator import (
    SingleTurnSample,
    EvaluationDataset,
    FaithfulnessMetric,
    AnswerRelevanceMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    RagasEvaluator,
    evaluate,
)

__all__ = [
    "RAGAS_STAGES",
    "get_completed_count",
    "get_progress_percentage",
    "SingleTurnSample",
    "EvaluationDataset",
    "FaithfulnessMetric",
    "AnswerRelevanceMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    "RagasEvaluator",
    "evaluate",
]
