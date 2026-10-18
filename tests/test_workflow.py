"""End-to-end flows through the real LangGraph (checkpointer, interrupt) in mock-LLM mode."""
import pytest

from app import db
from app.llm import LLMError, MockLLM
from app.service import SupportService


def steps(v):
    return [s["node"] for s in v["steps"]]


def test_small_refund_auto_resolves(svc):
    v = svc.start("The USB-C cable from ORD-1001 is defective. Please refund $24.99.", "alice@example.com")
    assert v["status"] == "completed" and v["outcome"] == "refunded"
    assert "approval" not in steps(v)
    assert steps(v)[:2] == ["triage", "retrieve"]
    assert v["triage"]["category"] == "product_defect"
    assert v["ticket"]["status"] == "resolved"
    assert "24.99" in v["reply"]
    assert db.list_refunds(svc.settings.db_path)[0]["amount"] == 24.99


def test_large_refund_pauses_then_approves(svc):
    v = svc.start("My headphones from ORD-1002 stopped working. I want a full refund.", "alice@example.com")
    assert v["status"] == "awaiting_approval"
    p = v["pending_approval"]
    assert p["amount"] == 189.0 and p["order_id"] == "ORD-1002"
    assert db.list_refunds(svc.settings.db_path) == []          # nothing released yet
    assert v["ticket"]["status"] == "pending_approval"
    assert [a["thread_id"] for a in svc.pending_approvals()] == [v["thread_id"]]

    v2 = svc.decide(v["thread_id"], True, "verified defect")
    assert v2["status"] == "completed" and v2["outcome"] == "refunded"
    assert db.list_refunds(svc.settings.db_path)[0]["amount"] == 189.0
    assert "approval" in steps(v2) and v2["ticket"]["status"] == "resolved"
    assert svc.pending_approvals() == []


def test_large_refund_rejected(svc):
    v = svc.start("My headphones from ORD-1002 stopped working. I want a full refund.", "alice@example.com")
    v2 = svc.decide(v["thread_id"], False, "outside policy")
    assert v2["status"] == "completed" and v2["outcome"] == "refund_rejected"
    assert db.list_refunds(svc.settings.db_path) == []
    assert v2["ticket"]["status"] == "refund_rejected" and "outside policy" in v2["reply"]


def test_cannot_decide_twice(svc):
    v = svc.start("My headphones from ORD-1002 stopped working. I want a full refund.", "alice@example.com")
    svc.decide(v["thread_id"], True)
    with pytest.raises(ValueError):
        svc.decide(v["thread_id"], True)


def test_pause_survives_restart(settings):
    a = SupportService(settings)
    v = a.start("My headphones from ORD-1002 stopped working. I want a full refund.", "alice@example.com")
    b = SupportService(settings)  # fresh process, same checkpoint DB
    assert b.view(v["thread_id"])["status"] == "awaiting_approval"
    assert b.decide(v["thread_id"], True)["outcome"] == "refunded"


def test_unknown_customer_escalates(svc):
    v = svc.start("I was charged twice for my last order!", "ghost@nowhere.com")
    assert v["status"] == "completed" and v["outcome"] == "escalated"
    assert v["ticket"]["status"] == "escalated" and "not found" in v["ticket"]["resolution"]
    assert db.list_refunds(svc.settings.db_path) == []


def test_high_urgency_refund_needs_approval_even_if_small(svc):
    v = svc.start("URGENT: my USB-C cable ORD-1001 is defective, refund $10 immediately", "alice@example.com")
    assert v["status"] == "awaiting_approval"
    assert "high-urgency" in v["pending_approval"]["why_approval_needed"]


def test_account_security_escalates(svc):
    v = svc.start("My account was hacked, unauthorized login!", "bob@example.com")
    assert v["outcome"] == "escalated"


def test_delivery_in_transit_just_informs(svc):
    v = svc.start("Where is my keyboard order ORD-1003?", "bob@example.com")
    assert v["outcome"] == "resolved" and "in transit" in v["reply"] and db.list_refunds(svc.settings.db_path) == []


def test_late_delivery_small_refund(svc):
    v = svc.start("My desk lamp ORD-1004 is very late", "bob@example.com")
    assert v["triage"]["category"] == "delivery" and v["outcome"] == "refunded"  # $45 <= $50, 12 days late


def test_old_defect_escalates_to_warranty(svc):
    v = svc.start("My espresso machine ORD-1005 is broken", "carol@example.com")
    assert v["outcome"] == "escalated" and "warranty" in v["ticket"]["resolution"]


def test_flaky_shipping_api_recovers_via_retry(svc):
    v = svc.start("Where is my smart watch order ORD-1006? Shipping tracking please", "dan@example.com")
    tool_step = next(s for s in v["steps"] if s["detail"].get("tool") == "get_order_status")
    assert tool_step["detail"]["result"]["attempts"] == 3 and v["outcome"] == "resolved"


def test_shipping_outage_gives_graceful_escalation(svc):
    v = svc.start("Where is my backpack order ORD-1007? delivery tracking", "dan@example.com")
    assert v["status"] == "completed" and v["outcome"] == "escalated"
    assert "human support team" in v["reply"]


def test_no_email_asks_for_it(svc):
    v = svc.start("My package never arrived", None)
    assert v["outcome"] == "resolved" and "email" in v["reply"]


class BrokenLLM(MockLLM):
    name = "broken"

    def triage(self, text):
        raise LLMError("boom")

    def chat(self, messages, tools, context):
        if not any(m["role"] == "tool" for m in messages):  # let ticket registration succeed via mock, then fail
            return super().chat(messages, tools, context)
        raise LLMError("provider down")


def test_llm_outage_falls_back_gracefully(settings):
    svc = SupportService(settings, llm=BrokenLLM())
    v = svc.start("The USB-C cable from ORD-1001 is defective. Please refund.", "alice@example.com")
    assert v["status"] == "completed" and v["outcome"] == "escalated"
    assert v["triage"]["category"] == "product_defect"   # rule-based triage fallback
    assert any("fallback" in s["title"].lower() for s in v["steps"])


class ValidationFlubLLM(MockLLM):
    """LLM that emits a bad tool call; validation error must come back as a tool result, not crash."""
    name = "flub"

    def chat(self, messages, tools, context):
        if not messages:
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "register_complaint", "arguments": '{"category":"x"}'}}]}
        return {"role": "assistant", "content": "done"}


def test_invalid_tool_args_do_not_crash(settings):
    svc = SupportService(settings, llm=ValidationFlubLLM())
    v = svc.start("something is broken with my order", "alice@example.com")
    assert v["status"] == "completed"
    tool = next(s for s in v["steps"] if s["node"] == "tools")
    assert tool["detail"]["result"]["error"] == "invalid_arguments"
