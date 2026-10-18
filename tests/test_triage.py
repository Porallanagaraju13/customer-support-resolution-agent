import pytest

from app.llm import MockLLM, rule_triage


@pytest.mark.parametrize("text,cat", [
    ("I was charged twice for my order", "billing"),
    ("My package never arrived, tracking says lost", "delivery"),
    ("The headphones are broken and stopped working", "product_defect"),
    ("I can't login, forgot my password", "account"),
])
def test_categories(text, cat):
    assert rule_triage(text).category == cat


@pytest.mark.parametrize("text,urg", [
    ("URGENT my account was hacked", "high"),
    ("I'm furious, this is unacceptable", "high"),
    ("Just wondering where my parcel is, no rush", "low"),
    ("My cable is faulty", "medium"),
])
def test_urgency(text, urg):
    assert rule_triage(text).urgency == urg


def test_mock_llm_triage_is_deterministic():
    m = MockLLM()
    assert m.triage("charged twice").model_dump() == m.triage("charged twice").model_dump()
