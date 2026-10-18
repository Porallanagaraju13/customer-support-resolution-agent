"""LangGraph multi-agent workflow.

START -> triage -> retrieve -> agent <-> tools
                                 |  \\-> approval (interrupt) -> tools
                                 \\-> finalize -> END
"""
from __future__ import annotations

import json
import logging
import operator
import time
from datetime import datetime, timezone
from typing import Annotated, Any, Optional, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import db
from .config import Settings
from .kb import KnowledgeBase
from .llm import LLM, LLMError, Triage, rule_triage
from .logging_conf import log_event
from .tools import ToolRegistry, approval_reason, openai_tool_schemas

MAX_AGENT_TURNS = 6
FALLBACK_REPLY = ("Sorry - we're having technical difficulties resolving this right now. "
                  "Your request has been recorded and passed to a human agent who will contact you shortly.")


class AgentState(TypedDict, total=False):
    complaint: str
    customer_email: Optional[str]
    thread_id: str
    triage: dict
    context: list
    messages: Annotated[list, operator.add]
    steps: Annotated[list, operator.add]
    ticket_id: Optional[str]
    approval: Optional[dict]
    turns: int
    reply: str
    outcome: str


def step(node: str, title: str, **detail) -> dict:
    return {"node": node, "title": title, "detail": detail, "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}


def _tool_calls(state: AgentState) -> list[dict]:
    msgs = state.get("messages") or []
    return (msgs[-1].get("tool_calls") or []) if msgs else []


def _gated(state: AgentState, threshold: float) -> list[tuple[dict, str]]:
    urgency = state["triage"]["urgency"]
    out = []
    for tc in _tool_calls(state):
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            continue
        why = approval_reason(tc["function"]["name"], args, urgency, threshold)
        if why:
            out.append((tc, why))
    return out


def build_graph(settings: Settings, llm: LLM, tools: ToolRegistry, kb: KnowledgeBase,
                checkpointer: BaseCheckpointSaver):
    schemas = openai_tool_schemas()

    # ---- Agent 1: triage ----
    def triage_node(state: AgentState):
        try:
            t = llm.triage(state["complaint"])
            note = ""
        except LLMError as exc:
            t, note = rule_triage(state["complaint"]), f"LLM triage failed ({exc}); used rule-based fallback"
            log_event("triage_fallback", logging.WARNING, error=str(exc))
        log_event("triage", thread_id=state.get("thread_id"), category=t.category, urgency=t.urgency)
        return {"triage": t.model_dump(), "steps": [step("triage", f"Classified as {t.category} / {t.urgency} urgency",
                                                          category=t.category, urgency=t.urgency, reason=t.reason, note=note)]}

    # ---- Agent 2: retrieval (RAG) ----
    def retrieve_node(state: AgentState):
        q = f"{state['triage']['category'].replace('_', ' ')} {state['complaint']}"
        hits = kb.search(q, k=3)
        log_event("retrieve", thread_id=state.get("thread_id"), sources=[f"{h['source']}#{h['section']}" for h in hits], backend=kb.backend)
        return {"context": hits, "steps": [step("retrieve", f"Retrieved {len(hits)} policy passages ({kb.backend})",
                                                 passages=[{"source": h["source"], "section": h["section"], "score": h["score"]} for h in hits])]}

    # ---- Agent 3: resolution (LLM with tool calling) ----
    def agent_node(state: AgentState):
        turns = state.get("turns", 0) + 1
        policy_text = "\n".join(f"- [{h['source']} / {h['section']}] {h['text']}" for h in state.get("context", []))
        ctx = {"triage": state["triage"], "complaint": state["complaint"], "customer_email": state.get("customer_email"),
               "policy_text": policy_text, "thread_id": state.get("thread_id")}
        if turns > MAX_AGENT_TURNS:
            msg = {"role": "assistant", "content": FALLBACK_REPLY}
            return {"turns": turns, "messages": [msg], "steps": [step("agent", "Turn limit reached - fallback reply")]}
        msg, err = None, None
        for attempt in range(1, 3):  # one retry with backoff on LLM failure
            try:
                msg = llm.chat(state.get("messages") or [], schemas, ctx)
                break
            except LLMError as exc:
                err = str(exc)
                log_event("llm_error", logging.WARNING, attempt=attempt, error=err)
                time.sleep(settings.backoff_base_s * attempt)
        if msg is None:  # graceful degradation: escalate if we have a ticket, else apologise
            if state.get("ticket_id") and not any(m.get("name") == "escalate_to_human" for m in state.get("messages", [])):
                msg = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_fallback_escalate", "type": "function",
                    "function": {"name": "escalate_to_human", "arguments": json.dumps({"reason": f"LLM unavailable: {err}"[:290]})}}]}
            else:
                msg = {"role": "assistant", "content": FALLBACK_REPLY}
            st = step("agent", "LLM unavailable - using fallback", error=err)
        elif msg.get("tool_calls"):
            names = [t["function"]["name"] for t in msg["tool_calls"]]
            st = step("agent", f"Resolution agent requests tools: {', '.join(names)}", tools=names)
        else:
            st = step("agent", "Resolution agent drafted final reply")
        out: dict[str, Any] = {"turns": turns, "messages": [msg], "steps": [st]}
        if not msg.get("tool_calls"):
            out["reply"] = msg.get("content") or FALLBACK_REPLY
        return out

    def route_after_agent(state: AgentState) -> str:
        if not _tool_calls(state):
            return "finalize"
        if state.get("approval") is None and _gated(state, settings.refund_threshold):
            return "approval"
        return "tools"

    # ---- Human-in-the-loop ----
    def approval_node(state: AgentState):
        # NOTE: this node re-runs from the top when resumed, so everything before
        # interrupt() must be idempotent.
        tc, why = _gated(state, settings.refund_threshold)[0]
        args = json.loads(tc["function"]["arguments"])
        cust = db.find_customer(settings.db_path, state.get("customer_email")) if state.get("customer_email") else None
        payload = {"ticket_id": state.get("ticket_id"), "action": tc["function"]["name"], "order_id": args.get("order_id"),
                   "amount": args.get("amount"), "reason": args.get("reason"), "why_approval_needed": why,
                   "urgency": state["triage"]["urgency"], "category": state["triage"]["category"],
                   "customer": {"name": cust["name"], "email": cust["email"], "tier": cust["tier"]} if cust else None,
                   "complaint": state["complaint"]}
        if state.get("ticket_id"):
            db.update_ticket(settings.db_path, state["ticket_id"], status="pending_approval")
        decision = interrupt(payload)  # <- graph pauses here; checkpointed until resumed
        decision = decision if isinstance(decision, dict) else {"decision": str(decision)}
        approved = str(decision.get("decision", "")).lower() in ("approve", "approved", "yes", "true")
        log_event("approval_decision", thread_id=state.get("thread_id"), approved=approved, note=decision.get("note"))
        return {"approval": {"approved": approved, "note": decision.get("note") or "", "approver": decision.get("approver", "human")},
                "steps": [step("approval", f"Human {'APPROVED' if approved else 'REJECTED'} {tc['function']['name']}",
                               note=decision.get("note") or "", why=why)]}

    # ---- tool execution ----
    def tools_node(state: AgentState):
        gated = {tc["id"] for tc, _ in _gated(state, settings.refund_threshold)}
        approval = state.get("approval")
        ticket_id = state.get("ticket_id")
        msgs, steps, new_ticket = [], [], ticket_id
        for tc in _tool_calls(state):
            name, raw = tc["function"]["name"], tc["function"]["arguments"]
            if tc["id"] in gated and not (approval and approval["approved"]):
                result = {"ok": False, "error": "rejected_by_human", "note": (approval or {}).get("note", "")}
                if ticket_id:
                    db.update_ticket(settings.db_path, ticket_id, status="refund_rejected",
                                     resolution=f"Refund rejected by reviewer. {result['note']}".strip())
            else:
                try:
                    args = json.loads(raw or "{}")
                    if name in ("issue_refund", "escalate_to_human") and new_ticket:
                        args["ticket_id"] = new_ticket  # never trust an LLM-supplied ticket id
                    raw = json.dumps(args)
                except json.JSONDecodeError:
                    pass
                result = tools.execute(name, raw, {"thread_id": state.get("thread_id")})
                if tc["id"] in gated and result.get("ok"):  # tell the LLM a human already signed this off
                    result["approved_by_human"] = True
                    result["approver"] = approval.get("approver", "human")
                if name == "register_complaint" and result.get("ok"):
                    new_ticket = result["ticket_id"]
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "name": name, "content": json.dumps(result, default=str)})
            ok = "ok" if result.get("ok") else f"failed: {result.get('error')}"
            steps.append(step("tools", f"{name} -> {ok}", tool=name, args=_safe(raw), result=result))
        out: dict[str, Any] = {"messages": msgs, "steps": steps, "approval": None}
        if new_ticket != ticket_id:
            out["ticket_id"] = new_ticket
        return out

    # ---- finalize ----
    def finalize_node(state: AgentState):
        reply = state.get("reply") or FALLBACK_REPLY
        tid = state.get("ticket_id")
        outcome = "resolved"
        for m in state.get("messages", []):
            if m["role"] == "tool":
                r = json.loads(m["content"])
                if m["name"] == "escalate_to_human" and r.get("ok"):
                    outcome = "escalated"
                if m["name"] == "issue_refund":
                    outcome = "refunded" if r.get("ok") else ("refund_rejected" if r.get("error") == "rejected_by_human" else outcome)
        if tid:
            t = db.get_ticket(settings.db_path, tid)
            if t and t["status"] in ("open", "pending_approval"):
                db.update_ticket(settings.db_path, tid, status="resolved")
            db.update_ticket(settings.db_path, tid, resolution=(t or {}).get("resolution") or reply[:300])
        log_event("finalize", thread_id=state.get("thread_id"), ticket_id=tid, outcome=outcome)
        return {"outcome": outcome, "steps": [step("finalize", f"Case closed: {outcome}", ticket_id=tid)]}

    g = StateGraph(AgentState)
    g.add_node("triage", triage_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("agent", agent_node)
    g.add_node("approval", approval_node)
    g.add_node("tools", tools_node)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "triage")
    g.add_edge("triage", "retrieve")
    g.add_edge("retrieve", "agent")
    g.add_conditional_edges("agent", route_after_agent, {"finalize": "finalize", "approval": "approval", "tools": "tools"})
    g.add_edge("approval", "tools")
    g.add_edge("tools", "agent")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def _safe(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw
