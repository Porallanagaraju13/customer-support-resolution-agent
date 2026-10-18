"""SQLite persistence: seeded customers/orders, tickets, refunds."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers(
  id TEXT PRIMARY KEY, name TEXT, email TEXT UNIQUE, tier TEXT);
CREATE TABLE IF NOT EXISTS orders(
  id TEXT PRIMARY KEY, customer_id TEXT, item TEXT, total REAL, status TEXT,
  placed_at TEXT, delivered_at TEXT, expected_delivery TEXT, carrier TEXT, tracking TEXT);
CREATE TABLE IF NOT EXISTS tickets(
  id INTEGER PRIMARY KEY AUTOINCREMENT, customer_email TEXT, category TEXT, urgency TEXT,
  summary TEXT, status TEXT, resolution TEXT, thread_id TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS refunds(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT, order_id TEXT, amount REAL,
  reason TEXT, created_at TEXT, UNIQUE(ticket_id, order_id));
CREATE TABLE IF NOT EXISTS conversations(
  thread_id TEXT PRIMARY KEY, message TEXT, customer_email TEXT, created_at TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(path: str):
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _day(offset: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=offset)).date().isoformat()


def init_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as c:
        c.executescript(SCHEMA)
        if c.execute("SELECT COUNT(*) FROM customers").fetchone()[0]:
            return
        c.executemany("INSERT INTO customers VALUES(?,?,?,?)", [
            ("C001", "Alice Johnson", "alice@example.com", "gold"),
            ("C002", "Bob Smith", "bob@example.com", "standard"),
            ("C003", "Carol Diaz", "carol@example.com", "standard"),
            ("C004", "Dan Lee", "dan@example.com", "silver"),
        ])
        # id, customer, item, total, status, placed, delivered, expected, carrier, tracking
        c.executemany("INSERT INTO orders VALUES(?,?,?,?,?,?,?,?,?,?)", [
            ("ORD-1001", "C001", "USB-C Charging Cable", 24.99, "delivered", _day(-9), _day(-5), _day(-5), "UPS", "1Z999AA10001"),
            ("ORD-1002", "C001", "Noise-Cancelling Headphones", 189.00, "delivered", _day(-14), _day(-10), _day(-10), "FedEx", "FX772100211"),
            ("ORD-1003", "C002", "Mechanical Keyboard", 119.00, "in_transit", _day(-3), None, _day(2), "DHL", "DH55012003"),
            ("ORD-1004", "C002", "Desk Lamp", 45.00, "in_transit", _day(-20), None, _day(-12), "USPS", "US940010004"),
            ("ORD-1005", "C003", "Espresso Machine", 329.00, "delivered", _day(-95), _day(-90), _day(-90), "UPS", "1Z999AA10005"),
            ("ORD-1006", "C004", "Smart Watch", 149.00, "in_transit", _day(-4), None, _day(1), "FedEx", "FX772100216"),
            ("ORD-1007", "C004", "Travel Backpack", 79.00, "in_transit", _day(-6), None, _day(0), "DHL", "DH55012007"),
        ])


# ---- customers / orders -------------------------------------------------
def find_customer(path: str, email: str | None = None, customer_id: str | None = None) -> dict | None:
    with connect(path) as c:
        if email:
            row = c.execute("SELECT * FROM customers WHERE lower(email)=lower(?)", (email,)).fetchone()
        else:
            row = c.execute("SELECT * FROM customers WHERE id=?", (customer_id,)).fetchone()
        if not row:
            return None
        cust = dict(row)
        cust["orders"] = [dict(r) for r in c.execute(
            "SELECT id,item,total,status,delivered_at FROM orders WHERE customer_id=? ORDER BY placed_at DESC", (cust["id"],))]
        return cust


def get_order(path: str, order_id: str) -> dict | None:
    with connect(path) as c:
        row = c.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
        return dict(row) if row else None


def refunded_total(path: str, order_id: str) -> float:
    with connect(path) as c:
        return c.execute("SELECT COALESCE(SUM(amount),0) FROM refunds WHERE order_id=?", (order_id,)).fetchone()[0]


# ---- tickets ------------------------------------------------------------
def fmt_ticket(i: int) -> str:
    return f"TKT-{i:04d}"


def create_ticket(path: str, email: str | None, category: str, urgency: str, summary: str,
                  thread_id: str | None = None) -> str:
    with connect(path) as c:
        cur = c.execute(
            "INSERT INTO tickets(customer_email,category,urgency,summary,status,thread_id,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (email, category, urgency, summary, "open", thread_id, _now(), _now()))
        return fmt_ticket(cur.lastrowid)


def update_ticket(path: str, ticket_id: str, status: str | None = None, resolution: str | None = None) -> None:
    n = int(ticket_id.split("-")[1])
    with connect(path) as c:
        if status:
            c.execute("UPDATE tickets SET status=?, updated_at=? WHERE id=?", (status, _now(), n))
        if resolution is not None:
            c.execute("UPDATE tickets SET resolution=?, updated_at=? WHERE id=?", (resolution, _now(), n))


def _ticket_row(r) -> dict:
    d = dict(r)
    d["id"] = fmt_ticket(d["id"])
    return d


def list_tickets(path: str) -> list[dict]:
    with connect(path) as c:
        return [_ticket_row(r) for r in c.execute("SELECT * FROM tickets ORDER BY id DESC")]


def get_ticket(path: str, ticket_id: str) -> dict | None:
    try:
        n = int(ticket_id.split("-")[1])
    except (IndexError, ValueError):
        return None
    with connect(path) as c:
        r = c.execute("SELECT * FROM tickets WHERE id=?", (n,)).fetchone()
        return _ticket_row(r) if r else None


def add_refund(path: str, ticket_id: str, order_id: str, amount: float, reason: str) -> tuple[int, bool]:
    """Idempotent per (ticket, order). Returns (refund_id, created)."""
    with connect(path) as c:
        row = c.execute("SELECT id FROM refunds WHERE ticket_id=? AND order_id=?", (ticket_id, order_id)).fetchone()
        if row:
            return row["id"], False
        cur = c.execute("INSERT INTO refunds(ticket_id,order_id,amount,reason,created_at) VALUES(?,?,?,?,?)",
                        (ticket_id, order_id, amount, reason, _now()))
        return cur.lastrowid, True


def existing_refund(path: str, ticket_id: str, order_id: str) -> dict | None:
    with connect(path) as c:
        r = c.execute("SELECT * FROM refunds WHERE ticket_id=? AND order_id=?", (ticket_id, order_id)).fetchone()
        return dict(r) if r else None


def list_refunds(path: str) -> list[dict]:
    with connect(path) as c:
        return [dict(r) for r in c.execute("SELECT * FROM refunds ORDER BY id DESC")]


# ---- conversations ------------------------------------------------------
def add_conversation(path: str, thread_id: str, message: str, email: str | None) -> None:
    with connect(path) as c:
        c.execute("INSERT INTO conversations VALUES(?,?,?,?)", (thread_id, message, email, _now()))


def list_conversations(path: str) -> list[dict]:
    with connect(path) as c:
        return [dict(r) for r in c.execute("SELECT * FROM conversations ORDER BY created_at DESC, rowid DESC")]
