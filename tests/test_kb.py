from app.kb import KnowledgeBase


def test_retrieval_finds_right_policy():
    kb = KnowledgeBase()
    assert kb.search("refund approval limit above 50 USD")[0]["source"] == "refund_policy.md"
    assert kb.search("how long does standard shipping take")[0]["source"] == "shipping_times.md"
    assert kb.search("warranty coverage 12 months")[0]["source"] == "warranty.md"


def test_embedding_failure_falls_back(monkeypatch):
    from app.config import Settings
    kb = KnowledgeBase(Settings(embeddings="openai", openai_api_key="bad", base_url="http://127.0.0.1:1"))
    assert kb.backend == "tfidf-local"
    assert kb.search("refund policy")
