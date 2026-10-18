import json

import pytest

from app import db
from app.tools import ToolRegistry, approval_reason, openai_tool_schemas


@pytest.fixture
def tools(settings):
    db.init_db(settings.db_path)
    return ToolRegistry(settings)


def test_register_and_get_customer(tools, settings):
    r = tools.execute("register_complaint", {"customer_email": "alice@example.com", "category": "billing",
                                             "urgency": "low", "summary": "double charge"})
    assert r["ok"] and r["ticket_id"] == "TKT-0001"
    assert db.get_ticket(settings.db_path, "TKT-0001")["status"] == "open"
    c = tools.execute("get_customer", {"email": "ALICE@example.com"})
    assert c["ok"] and c["customer"]["name"] == "Alice Johnson" and len(c["customer"]["orders"]) == 2


def test_unknown_customer(tools):
    r = tools.execute("get_customer", {"email": "nobody@example.com"})
    assert not r["ok"] and "customer_not_found" in r["detail"]


@pytest.mark.parametrize("name,args", [
    ("get_order_status", {"order_id": "12345"}),
    ("issue_refund", {"order_id": "ORD-1001", "amount": -5, "reason": "nope"}),
    ("issue_refund", {"order_id": "ORD-1001", "amount": 5}),
    ("get_customer", {}),
    ("get_customer", {"email": "not-an-email"}),
    ("register_complaint", {"category": "weird", "urgency": "low", "summary": "x y z"}),
])
def test_pydantic_validation_rejects_bad_args(tools, name, args):
    r = tools.execute(name, args)
    assert not r["ok"] and r["error"] == "invalid_arguments"


def test_bad_json_and_unknown_tool(tools):
    assert tools.execute("get_customer", "{oops")["error"] == "invalid_arguments"
    assert tools.execute("drop_tables", {})["error"] == "unknown_tool"


def test_retry_with_backoff_recovers(tools):
    r = tools.execute("get_order_status", {"order_id": "ORD-1006"})  # fails twice then ok
    assert r["ok"] and r["attempts"] == 3 and r["shipment"]["carrier"] == "FedEx"


def test_retries_exhausted_graceful_error(tools):
    r = tools.execute("get_order_status", {"order_id": "ORD-1007"})
    assert not r["ok"] and r["error"] == "unavailable" and r["attempts"] == 3


def test_timeout(settings):
    import time
    from dataclasses import replace
    s = replace(settings, tool_timeout_s=0.05, tool_max_retries=2)
    t = ToolRegistry(s)
    db.init_db(s.db_path)
    t.handlers["get_order_status"] = lambda a, c: time.sleep(0.5)
    r = t.execute("get_order_status", {"order_id": "ORD-1001"})
    assert not r["ok"] and r["error"] == "unavailable" and r["attempts"] == 2


def test_refund_is_bounded_and_idempotent(tools, settings):
    tools.execute("register_complaint", {"category": "billing", "urgency": "low", "summary": "refund please"})
    too_much = tools.execute("issue_refund", {"ticket_id": "TKT-0001", "order_id": "ORD-1001", "amount": 100, "reason": "test"})
    assert not too_much["ok"] and "exceeds refundable" in too_much["detail"]
    ok = tools.execute("issue_refund", {"ticket_id": "TKT-0001", "order_id": "ORD-1001", "amount": 24.99, "reason": "test"})
    again = tools.execute("issue_refund", {"ticket_id": "TKT-0001", "order_id": "ORD-1001", "amount": 24.99, "reason": "test"})
    assert ok["ok"] and again["duplicate"] and len(db.list_refunds(settings.db_path)) == 1
    assert db.get_ticket(settings.db_path, "TKT-0001")["status"] == "resolved"


def test_escalate_updates_ticket(tools, settings):
    tools.execute("register_complaint", {"category": "account", "urgency": "high", "summary": "hacked account"})
    assert tools.execute("escalate_to_human", {"ticket_id": "TKT-0001", "reason": "security"})["ok"]
    assert db.get_ticket(settings.db_path, "TKT-0001")["status"] == "escalated"


def test_approval_policy():
    assert approval_reason("issue_refund", {"amount": 50}, "medium", 50) is None
    assert approval_reason("issue_refund", {"amount": 50.01}, "medium", 50)
    assert approval_reason("issue_refund", {"amount": 5}, "high", 50)
    assert approval_reason("escalate_to_human", {}, "high", 50) is None


def test_openai_tool_schemas():
    names = {t["function"]["name"] for t in openai_tool_schemas()}
    assert names == {"register_complaint", "get_customer", "get_order_status", "issue_refund", "escalate_to_human"}
    json.dumps(openai_tool_schemas())
