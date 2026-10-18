"""Resolution-agent tools: Pydantic-validated args, retries with backoff, timeouts."""
from __future__ import annotations

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import date
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from . import db
from .config import Settings
from .logging_conf import log_event

log = logging.getLogger("support_agent.tools")

Category = Literal["billing", "delivery", "product_defect", "account"]
Urgency = Literal["low", "medium", "high"]


class TransientToolError(Exception):
    """Retryable failure (network blip, 5xx, timeout)."""


class ToolError(Exception):
    """Permanent, non-retryable failure (bad business state)."""


# ---- argument schemas ---------------------------------------------------
class RegisterComplaintArgs(BaseModel):
    customer_email: Optional[str] = Field(None, description="Customer email if known")
    category: Category
    urgency: Urgency
    summary: str = Field(min_length=3, max_length=500)


class GetCustomerArgs(BaseModel):
    email: Optional[str] = None
    customer_id: Optional[str] = None

    @field_validator("email")
    @classmethod
    def _email(cls, v):
        if v is not None and not ("@" in v and "." in v.split("@")[-1]):
            raise ValueError("invalid email address")
        return v

    def model_post_init(self, _ctx) -> None:
        if not (self.email or self.customer_id):
            raise ValueError("provide email or customer_id")


class GetOrderStatusArgs(BaseModel):
    order_id: str = Field(pattern=r"^ORD-\d{4,8}$")


class IssueRefundArgs(BaseModel):
    ticket_id: Optional[str] = Field(None, description="Filled in by the system")
    order_id: str = Field(pattern=r"^ORD-\d{4,8}$")
    amount: float = Field(gt=0, le=10000)
    reason: str = Field(min_length=3, max_length=300)


class EscalateArgs(BaseModel):
    ticket_id: Optional[str] = Field(None, description="Filled in by the system")
    reason: str = Field(min_length=3, max_length=300)


TOOL_SPECS = {
    "register_complaint": ("Create a support ticket for the complaint. Call this first.", RegisterComplaintArgs),
    "get_customer": ("Look up a customer and their orders by email or customer_id.", GetCustomerArgs),
    "get_order_status": ("Get live shipping status for an order from the carrier API.", GetOrderStatusArgs),
    "issue_refund": ("Refund an order. Refunds above the threshold need human approval (handled by the system).", IssueRefundArgs),
    "escalate_to_human": ("Hand the case to a human agent when it cannot be resolved automatically.", EscalateArgs),
}
GATED_TOOLS = {"issue_refund"}


def openai_tool_schemas() -> list[dict]:
    out = []
    for name, (desc, model) in TOOL_SPECS.items():
        schema = model.model_json_schema()
        schema.pop("title", None)
        out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": schema}})
    return out


def approval_reason(name: str, args: dict, urgency: str, threshold: float) -> str | None:
    """Return why a tool call needs human approval, or None if it can run automatically."""
    if name not in GATED_TOOLS:
        return None
    try:
        amount = float(args.get("amount", 0))
    except (TypeError, ValueError):
        amount = 0.0
    if amount > threshold:
        return f"refund of ${amount:.2f} exceeds the ${threshold:.2f} auto-approval limit"
    if urgency == "high":
        return "high-urgency case requires human sign-off"
    return None


# ---- mock shipping API --------------------------------------------------
class MockShippingAPI:
    """Stand-in for a carrier API. ORD-1006 fails twice then recovers (retry demo);
    ORD-1007 is always down (graceful-fallback demo)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.calls: dict[str, int] = {}

    def fetch(self, order_id: str) -> dict:
        n = self.calls[order_id] = self.calls.get(order_id, 0) + 1
        if order_id == "ORD-1007":
            raise TransientToolError("carrier API 503 Service Unavailable")
        if order_id == "ORD-1006" and n <= 2:
            raise TransientToolError("carrier API timeout")
        o = db.get_order(self.db_path, order_id)
        if not o:
            raise ToolError(f"order {order_id} not found")
        days_late = 0
        if o["status"] != "delivered" and o["expected_delivery"]:
            days_late = max(0, (date.today() - date.fromisoformat(o["expected_delivery"])).days)
        days_since_delivery = None
        if o["delivered_at"]:
            days_since_delivery = (date.today() - date.fromisoformat(o["delivered_at"])).days
        return {"order_id": order_id, "item": o["item"], "total": o["total"], "status": o["status"],
                "carrier": o["carrier"], "tracking": o["tracking"],
                "expected_delivery": o["expected_delivery"], "delivered_at": o["delivered_at"],
                "days_late": days_late, "days_since_delivery": days_since_delivery}


# ---- registry -----------------------------------------------------------
class ToolRegistry:
    def __init__(self, settings: Settings):
        self.s = settings
        self.shipping = MockShippingAPI(settings.db_path)
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")
        self.handlers: dict[str, Callable] = {
            "register_complaint": self._register_complaint,
            "get_customer": self._get_customer,
            "get_order_status": self._get_order_status,
            "issue_refund": self._issue_refund,
            "escalate_to_human": self._escalate,
        }

    # public entry point: never raises, always returns a JSON-able dict
    def execute(self, name: str, raw_args, context: dict | None = None) -> dict:
        context = context or {}
        if name not in TOOL_SPECS:
            return {"ok": False, "error": "unknown_tool", "detail": f"no tool named {name}"}
        try:
            data = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
            args = TOOL_SPECS[name][1](**data)
        except (json.JSONDecodeError, TypeError) as exc:
            return {"ok": False, "error": "invalid_arguments", "detail": f"bad JSON: {exc}"}
        except ValidationError as exc:
            msgs = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            log_event("tool_validation_failed", logging.WARNING, tool=name, detail=msgs)
            return {"ok": False, "error": "invalid_arguments", "detail": msgs}
        except ValueError as exc:
            return {"ok": False, "error": "invalid_arguments", "detail": str(exc)}

        handler = self.handlers[name]
        attempts = 0
        last = ""
        for attempt in range(1, self.s.tool_max_retries + 1):
            attempts = attempt
            t0 = time.time()
            try:
                fut = self._pool.submit(handler, args, context)
                result = fut.result(timeout=self.s.tool_timeout_s)
                log_event("tool_ok", tool=name, attempt=attempt, ms=int((time.time() - t0) * 1000))
                return {"ok": True, "attempts": attempts, **result}
            except ToolError as exc:
                log_event("tool_failed_permanent", logging.WARNING, tool=name, error=str(exc))
                return {"ok": False, "error": "tool_error", "detail": str(exc), "attempts": attempts}
            except (TransientToolError, FutureTimeout) as exc:
                last = str(exc) or "timeout"
                log_event("tool_transient_error", logging.WARNING, tool=name, attempt=attempt, error=last)
                if attempt < self.s.tool_max_retries:
                    delay = self.s.backoff_base_s * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
                    time.sleep(delay)
            except Exception as exc:  # unexpected bug: do not crash the graph
                log.exception("tool_crashed", extra={"tool": name})
                return {"ok": False, "error": "internal_error", "detail": str(exc), "attempts": attempts}
        return {"ok": False, "error": "unavailable",
                "detail": f"{name} failed after {attempts} attempts: {last}", "attempts": attempts}

    # ---- handlers ----
    def _register_complaint(self, a: RegisterComplaintArgs, ctx: dict) -> dict:
        tid = db.create_ticket(self.s.db_path, a.customer_email, a.category, a.urgency, a.summary, ctx.get("thread_id"))
        return {"ticket_id": tid}

    def _get_customer(self, a: GetCustomerArgs, ctx: dict) -> dict:
        c = db.find_customer(self.s.db_path, a.email, a.customer_id)
        if not c:
            raise ToolError("customer_not_found")
        return {"customer": c}

    def _get_order_status(self, a: GetOrderStatusArgs, ctx: dict) -> dict:
        return {"shipment": self.shipping.fetch(a.order_id)}

    def _issue_refund(self, a: IssueRefundArgs, ctx: dict) -> dict:
        if not a.ticket_id:
            raise ToolError("ticket_id required; register the complaint first")
        order = db.get_order(self.s.db_path, a.order_id)
        if not order:
            raise ToolError(f"order {a.order_id} not found")
        existing = db.existing_refund(self.s.db_path, a.ticket_id, a.order_id)
        if existing:  # idempotent: a retried call must not double-refund
            return {"refund_id": f"RF-{existing['id']:04d}", "amount": existing["amount"], "order_id": a.order_id, "duplicate": True}
        remaining = round(order["total"] - db.refunded_total(self.s.db_path, a.order_id), 2)
        if a.amount > remaining + 1e-9:
            raise ToolError(f"amount ${a.amount:.2f} exceeds refundable balance ${remaining:.2f}")
        rid, created = db.add_refund(self.s.db_path, a.ticket_id, a.order_id, a.amount, a.reason)
        db.update_ticket(self.s.db_path, a.ticket_id, status="resolved", resolution=f"Refunded ${a.amount:.2f} for {a.order_id}")
        return {"refund_id": f"RF-{rid:04d}", "amount": a.amount, "order_id": a.order_id, "duplicate": not created}

    def _escalate(self, a: EscalateArgs, ctx: dict) -> dict:
        if a.ticket_id:
            db.update_ticket(self.s.db_path, a.ticket_id, status="escalated", resolution=f"Escalated: {a.reason}")
        return {"queue": "human-support", "ticket_id": a.ticket_id, "reason": a.reason}
