"""Application service: owns the compiled graph, runs threads in the background."""
from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from . import db
from .config import Settings
from .graph import build_graph
from .kb import KnowledgeBase
from .llm import LLM, build_llm
from .logging_conf import log_event
from .tools import ToolRegistry


class SupportService:
    def __init__(self, settings: Settings | None = None, llm: LLM | None = None):
        self.settings = settings or Settings.from_env()
        Path(self.settings.data_dir).mkdir(parents=True, exist_ok=True)
        db.init_db(self.settings.db_path)
        self.llm = llm or build_llm(self.settings)
        self.kb = KnowledgeBase(self.settings)
        self.tools = ToolRegistry(self.settings)
        conn = sqlite3.connect(self.settings.checkpoint_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(conn)
        self.graph = build_graph(self.settings, self.llm, self.tools, self.kb, self.checkpointer)
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="run")
        self._inflight: dict[str, Future] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()

    # ---- running ----
    def _cfg(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _run(self, thread_id: str, payload) -> None:
        try:
            self.graph.invoke(payload, self._cfg(thread_id))
        except Exception as exc:  # never lose the failure silently
            logging.getLogger("support_agent").exception("run_failed", extra={"thread_id": thread_id})
            self._errors[thread_id] = str(exc)

    def _submit(self, thread_id: str, payload, wait: bool, timeout: float = 120) -> dict:
        with self._lock:
            fut = self._pool.submit(self._run, thread_id, payload)
            self._inflight[thread_id] = fut
        if wait:
            fut.result(timeout=timeout)
        return self.view(thread_id)

    def start(self, message: str, customer_email: str | None = None, wait: bool = True) -> dict:
        thread_id = uuid.uuid4().hex[:12]
        db.add_conversation(self.settings.db_path, thread_id, message, customer_email)
        log_event("conversation_started", thread_id=thread_id, customer_email=customer_email)
        init = {"complaint": message, "customer_email": customer_email, "thread_id": thread_id,
                "messages": [], "steps": [], "turns": 0, "approval": None}
        return self._submit(thread_id, init, wait)

    def decide(self, thread_id: str, approve: bool, note: str = "", approver: str = "human", wait: bool = True) -> dict:
        v = self.view(thread_id)
        if v["status"] != "awaiting_approval":
            raise ValueError(f"thread {thread_id} is not awaiting approval (status={v['status']})")
        log_event("approval_submitted", thread_id=thread_id, approve=approve, approver=approver)
        cmd = Command(resume={"decision": "approve" if approve else "reject", "note": note, "approver": approver})
        return self._submit(thread_id, cmd, wait)

    # ---- reading ----
    def view(self, thread_id: str) -> dict:
        snap = self.graph.get_state(self._cfg(thread_id))
        fut = self._inflight.get(thread_id)
        running = bool(fut and not fut.done())
        vals = snap.values or {}
        pending = None
        for t in snap.tasks:
            if t.interrupts:
                pending = t.interrupts[0].value
        if thread_id in self._errors:
            status = "error"
        elif running or (not snap.values):
            status = "running"
        elif pending and snap.next:
            status = "awaiting_approval"
        elif snap.next:
            status = "running" if running else "interrupted"
        else:
            status = "completed"
        ticket = db.get_ticket(self.settings.db_path, vals["ticket_id"]) if vals.get("ticket_id") else None
        return {"thread_id": thread_id, "status": status, "complaint": vals.get("complaint"),
                "triage": vals.get("triage"), "steps": vals.get("steps", []), "reply": vals.get("reply"),
                "outcome": vals.get("outcome"), "ticket": ticket,
                "pending_approval": pending if status == "awaiting_approval" else None,
                "error": self._errors.get(thread_id)}

    def exists(self, thread_id: str) -> bool:
        return any(c["thread_id"] == thread_id for c in db.list_conversations(self.settings.db_path))

    def pending_approvals(self) -> list[dict]:
        out = []
        for c in db.list_conversations(self.settings.db_path):
            v = self.view(c["thread_id"])
            if v["status"] == "awaiting_approval":
                out.append({"thread_id": c["thread_id"], "created_at": c["created_at"], **v["pending_approval"]})
        return out

    def threads(self) -> list[dict]:
        return [{**c, "status": self.view(c["thread_id"])["status"]} for c in db.list_conversations(self.settings.db_path)]
