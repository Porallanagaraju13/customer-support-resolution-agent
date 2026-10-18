"""FastAPI app: REST API + static single-page UI."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import db
from .config import Settings
from .logging_conf import setup_logging
from .service import SupportService

STATIC = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str = Field(min_length=3, max_length=2000)
    customer_email: Optional[str] = None
    wait: bool = Field(True, description="Block until the run finishes or pauses for approval")


class ApprovalRequest(BaseModel):
    decision: Literal["approve", "reject"]
    note: str = ""
    approver: str = "support-lead"
    wait: bool = True


def create_app(settings: Settings | None = None, service: SupportService | None = None) -> FastAPI:
    setup_logging()
    svc = service or SupportService(settings)
    app = FastAPI(title="Customer Support Resolution Agent", version="1.0.0")
    app.state.service = svc

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/config")
    def config():
        s = svc.settings
        return {"llm_mode": s.effective_mode, "llm": svc.llm.name, "retrieval": svc.kb.backend,
                "refund_approval_threshold": s.refund_threshold}

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        return svc.start(req.message, req.customer_email, req.wait)

    @app.get("/api/threads")
    def threads():
        return svc.threads()

    @app.get("/api/threads/{thread_id}")
    def thread(thread_id: str):
        if not svc.exists(thread_id):
            raise HTTPException(404, "thread not found")
        return svc.view(thread_id)

    @app.get("/api/approvals")
    def approvals():
        return svc.pending_approvals()

    @app.post("/api/approvals/{thread_id}")
    def decide(thread_id: str, req: ApprovalRequest):
        if not svc.exists(thread_id):
            raise HTTPException(404, "thread not found")
        try:
            return svc.decide(thread_id, req.decision == "approve", req.note, req.approver, req.wait)
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.get("/api/tickets")
    def tickets():
        return db.list_tickets(svc.settings.db_path)

    @app.get("/api/tickets/{ticket_id}")
    def ticket(ticket_id: str):
        t = db.get_ticket(svc.settings.db_path, ticket_id)
        if not t:
            raise HTTPException(404, "ticket not found")
        return t

    @app.get("/api/refunds")
    def refunds():
        return db.list_refunds(svc.settings.db_path)

    return app
