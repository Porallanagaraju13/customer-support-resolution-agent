"""Demo sample data: keeps the hosted demo's tickets table populated.

Runs the real graph (mock LLM) so every sample is genuine: real tickets, refunds and checkpoints, including one
thread that is truly paused at the approval interrupt (Approve/Reject work). Everything is tagged "[Sample]" and
uses example.com addresses only. Only runs when DEMO_SEED is on and the tickets table is empty.
"""
from __future__ import annotations

import logging

from . import db

TAG = "[Sample]"

# (message, customer email, approve-at-seed-time: None = leave as is, True = approve)
SAMPLES = [
    ("The Mechanical Keyboard from ORD-1003 hasn't arrived yet, where is my delivery?", "bob@example.com", None),
    ("I can't log in to my account, the password reset email never arrives.", "carol@example.com", None),
    ("I was charged twice for my order and want a refund.", "ghost.user@example.com", None),  # unknown -> escalated
    ("The Standing Desk from ORD-1008 has a cracked frame and is defective. I want a full refund.",
     "erin@example.com", True),  # large refund, human approved
    ("The USB-C cable from ORD-1001 is defective. Please refund $24.99.", "alice@example.com", None),  # small, auto
    ("My headphones from ORD-1002 stopped working. I want a full refund.", "alice@example.com", None),  # pending
]


def _ensure_sample_customer(path: str) -> None:
    with db.connect(path) as c:
        c.execute("INSERT OR IGNORE INTO customers VALUES(?,?,?,?)", ("C005", "Erin Park", "erin@example.com", "standard"))
        c.execute("INSERT OR IGNORE INTO orders VALUES(?,?,?,?,?,?,?,?,?,?)",
                  ("ORD-1008", "C005", "Standing Desk", 259.00, "delivered", db._day(-12), db._day(-8), db._day(-8),
                   "UPS", "1Z999AA10008"))


def seed_demo_data(svc) -> bool:
    path = svc.settings.db_path
    if db.list_tickets(path):
        return False
    _ensure_sample_customer(path)
    for msg, email, approve in SAMPLES:
        v = svc.start(f"{TAG} {msg}", email, True)
        if approve and v["status"] == "awaiting_approval":
            svc.decide(v["thread_id"], True, "Sample data: verified defect, approved by support lead", "support-lead")
    logging.getLogger("support_agent").info("demo_seed_done")
    return True
