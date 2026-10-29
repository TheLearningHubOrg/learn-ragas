"""
The Learning Hub - Ragas Automated RAG Evaluation Pedagogical Mastery Roadmap.
7-Stage progression covering decoupled metrics, mathematical formulas, synthetic testsets, and CI/CD quality gates.
"""

from typing import List, Dict, Any

RAGAS_STAGES: List[Dict[str, Any]] = [
    {
        "stage_number": 1,
        "title": "The RAG Evaluation Crisis & Decoupled Assessment",
        "domain_focus": "Why end-to-end evaluation fails, retrieval vs generation separation, and golden dataset creation.",
        "completed": True,
        "key_topics": ["Decoupled Component Assessment", "Triad of RAG Metrics", "Evaluation Bias & Variance", "Testset Ground Truth"]
    },
    {
        "stage_number": 2,
        "title": "Core Metric Formulations & Mathematical Foundations",
        "domain_focus": "Mathematical derivation of Faithfulness, Answer Relevance, Context Precision, and Context Recall.",
        "completed": True,
        "key_topics": ["Faithfulness (Claim Verification)", "Context Precision@K", "Context Recall (Sentence Attribution)", "Answer Relevance Cosine Score"]
    },
    {
        "stage_number": 3,
        "title": "End-to-End Generation & Semantic Similarity Metrics",
        "domain_focus": "Semantic similarity, answer correctness, BLEU/ROUGE comparison, and non-parametric LLM-as-a-judge.",
        "completed": True,
        "key_topics": ["Semantic Similarity", "Answer Correctness", "Judge Prompting Calibration", "Cross-Model Consensus"]
    },
    {
        "stage_number": 4,
        "title": "Agentic & Tool-Calling Multi-Hop Evaluation",
        "domain_focus": "Multi-hop query answering, tool selection precision, execution trajectory grading, and agent reflection.",
        "completed": True,
        "key_topics": ["Multi-Hop Context Reasoning", "Tool Call Correctness", "Intermediate Step Quality", "Agent Trajectory Grading"]
    },
    {
        "stage_number": 5,
        "title": "Synthetic Testset Generation & Knowledge Graph Evolution",
        "domain_focus": "TestsetGenerator, Evol-Instruct query transformations (reasoning, conditioning, multi-context), and distributions.",
        "completed": True,
        "key_topics": ["Evol-Instruct Paradigm", "Simple vs Complex Queries", "Corpus Knowledge Graphs", "Synthetic Query Filtering"]
    },
    {
        "stage_number": 6,
        "title": "Framework Integrations & CI/CD Quality Gates",
        "domain_focus": "LangChain and LlamaIndex evaluators, GitHub Actions automated gates, threshold alerts, and regression testing.",
        "completed": True,
        "key_topics": ["CI/CD Pull Request Quality Gates", "Pass/Fail Threshold Matrix", "Regression Detection", "Automated Scorecards"]
    },
    {
        "stage_number": 7,
        "title": "Staff AI Evaluation Architect Handbook & Production Cheatsheet",
        "domain_focus": "Staff architecture trade-offs, token economics of evaluation, asynchronous batching, and production telemetry.",
        "completed": True,
        "key_topics": ["Cost-Effective Sampling Strategies", "Evaluation Async Concurrency", "Continuous Live Auditing", "Production Cheatsheet"]
    }
]

def get_completed_count() -> int:
    return sum(1 for s in RAGAS_STAGES if s["completed"])

def get_progress_percentage() -> int:
    return round((get_completed_count() / len(RAGAS_STAGES)) * 100)
