"""
Unit test suite for the Ragas Automated RAG Evaluation Engine.
Tests cover Faithfulness, Answer Relevance, Context Precision@K, Context Recall, and CI/CD Quality Gates.
"""

import unittest
from src.learn_ragas.evaluator import (
    SingleTurnSample,
    EvaluationDataset,
    FaithfulnessMetric,
    AnswerRelevanceMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    evaluate
)
from src.learn_ragas.roadmap import (
    RAGAS_STAGES,
    get_completed_count,
    get_progress_percentage
)


class TestRagasEngine(unittest.TestCase):

    def test_faithfulness_metric_grounded_vs_hallucinated(self):
        metric = FaithfulnessMetric()

        # Completely grounded
        sample_grounded = SingleTurnSample(
            user_input="What is Redis?",
            retrieved_contexts=["Redis is an in-memory key-value data store."],
            response="Redis is an in-memory key-value data store."
        )
        self.assertEqual(metric.score(sample_grounded), 1.0)

        # Partially hallucinated
        sample_hallucinated = SingleTurnSample(
            user_input="What is Redis?",
            retrieved_contexts=["Redis is an in-memory key-value data store."],
            response="Redis is an in-memory store. It was written in Rust and uses multi-raft distributed consensus."
        )
        score = metric.score(sample_hallucinated)
        self.assertLess(score, 1.0)

    def test_answer_relevance_metric(self):
        metric = AnswerRelevanceMetric()

        sample_relevant = SingleTurnSample(
            user_input="How does Kubernetes schedule pods onto nodes?",
            retrieved_contexts=["Kube-scheduler assigns pods based on resource filtering and scoring."],
            response="Kubernetes schedule pods onto nodes using kube-scheduler filtering and scoring algorithms."
        )
        score_high = metric.score(sample_relevant)
        self.assertGreater(score_high, 0.4)

        sample_irrelevant = SingleTurnSample(
            user_input="How does Kubernetes schedule pods?",
            retrieved_contexts=["Context text"],
            response="The weather today is cloudy with light rain."
        )
        score_low = metric.score(sample_irrelevant)
        self.assertLess(score_low, 0.2)

    def test_context_precision_ranking(self):
        metric = ContextPrecisionMetric()

        # Relevant chunk at rank 1 -> perfect precision
        sample_good_rank = SingleTurnSample(
            user_input="explain raft consensus",
            retrieved_contexts=[
                "Raft consensus handles leader election and log replication.",  # Rank 1: relevant
                "Bananas are a great source of potassium."  # Rank 2: irrelevant
            ],
            response="Raft is a consensus algorithm.",
            reference="Raft consensus protocol coordinates leader election."
        )
        score_good = metric.score(sample_good_rank)
        self.assertEqual(score_good, 1.0)

        # Irrelevant chunk at rank 1 -> degraded precision
        sample_poor_rank = SingleTurnSample(
            user_input="explain raft consensus",
            retrieved_contexts=[
                "Bananas are a great source of potassium.",  # Rank 1: distractor
                "Raft consensus handles leader election and log replication."  # Rank 2: relevant
            ],
            response="Raft is a consensus algorithm.",
            reference="Raft consensus protocol coordinates leader election."
        )
        score_poor = metric.score(sample_poor_rank)
        self.assertEqual(score_poor, 0.5)

    def test_context_recall_metric(self):
        metric = ContextRecallMetric()

        sample = SingleTurnSample(
            user_input="query",
            retrieved_contexts=["PostgreSQL uses MVCC for transactions. It writes changes to WAL."],
            response="answer",
            reference="PostgreSQL uses MVCC for transactions. It writes changes to WAL."
        )
        self.assertEqual(metric.score(sample), 1.0)

    def test_quality_gate_checks(self):
        sample = SingleTurnSample(
            user_input="test query",
            retrieved_contexts=["test query context data"],
            response="test query context data response",
            reference="test query context data"
        )
        dataset = EvaluationDataset([sample])
        result = evaluate(dataset)

        passed, failures = result.passes_quality_gate({"faithfulness": 0.50, "context_recall": 0.50})
        self.assertTrue(passed)
        self.assertEqual(len(failures), 0)

        # Trigger failure with impossible threshold
        passed_strict, failures_strict = result.passes_quality_gate({"faithfulness": 1.01})
        self.assertFalse(passed_strict)
        self.assertTrue(len(failures_strict) >= 1)

    def test_roadmap_7_stages(self):
        self.assertEqual(len(RAGAS_STAGES), 7)
        self.assertEqual(get_completed_count(), 7)
        self.assertEqual(get_progress_percentage(), 100)


if __name__ == "__main__":
    unittest.main()
