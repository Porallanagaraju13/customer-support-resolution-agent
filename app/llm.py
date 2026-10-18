"""LLM abstraction: OpenAI-compatible client + deterministic offline MockLLM."""
from __future__ import annotations

import json
import re
import time
from typing import Literal, Protocol

from pydantic import BaseModel, ValidationError

from .config import Settings

CATEGORIES = ("billing", "delivery", "product_defect", "account")


class Triage(BaseModel):
    category: Literal["billing", "delivery", "product_defect", "account"]
    urgency: Literal["low", "medium", "high"]
    reason: str = ""


class LLMError(Exception):
    pass


class LLM(Protocol):
    name: str

    def triage(self, text: str) -> Triage: ...

    def chat(self, messages: list[dict], tools: list[dict], context: dict) -> dict: ...


# ---------------------------------------------------------------------------
# Rule-based triage (also used as the fallback when a real LLM fails)
# ---------------------------------------------------------------------------
_CAT_KEYWORDS = {
    "billing": ["charge", "charged", "billing", "invoice", "double", "twice", "payment", "overcharged", "price"],
    "delivery": ["deliver", "delivery", "shipping", "shipped", "package", "parcel", "arrive", "arrived",
                 "tracking", "late", "lost", "where is", "transit", "never received"],
    "product_defect": ["broken", "defect", "defective", "stopped working", "not working", "doesn't work",
                       "does not work", "damaged", "faulty", "cracked", "malfunction", "dead on arrival"],
    "account": ["password", "login", "log in", "sign in", "account", "locked", "hacked", "username", "2fa"],
}
_HIGH = ["urgent", "asap", "immediately", "fraud", "hacked", "unauthorized", "unauthorised", "furious",
         "lawyer", "legal", "chargeback", "emergency", "outraged", "unacceptable"]
_LOW = ["no rush", "whenever", "just wondering", "just curious", "when you get a chance", "quick question"]


def rule_triage(text: str) -> Triage:
    t = text.lower()
    scores = {c: sum(1 for k in kws if k in t) for c, kws in _CAT_KEYWORDS.items()}
    # defect/billing language is more specific than generic 'account'/'delivery' words
    prio = ["product_defect", "billing", "delivery", "account"]
    best = max(prio, key=lambda c: (scores[c], -prio.index(c)))
    if scores[best] == 0:
        best = "account"
    urgency = "high" if any(k in t for k in _HIGH) else "low" if any(k in t for k in _LOW) else "medium"
    return Triage(category=best, urgency=urgency, reason=f"keyword match ({scores[best]} hits)")


# ---------------------------------------------------------------------------
# Mock LLM: deterministic policy-following stub that speaks the tool-calling protocol
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
ORDER_RE = re.compile(r"ORD-\d{4,8}", re.I)
MONEY_RE = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)")


def _call(i: int, name: str, **args) -> dict:
    return {"id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class MockLLM:
    name = "mock-rule-based"

    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s

    def triage(self, text: str) -> Triage:
        self._sleep()
        return rule_triage(text)

    def _sleep(self):
        if self.delay_s:
            time.sleep(self.delay_s)

    def chat(self, messages: list[dict], tools: list[dict], context: dict) -> dict:
        self._sleep()
        results: dict[str, dict] = {}
        for m in messages:
            if m["role"] == "tool":
                results[m["name"]] = json.loads(m["content"])
        triage = context["triage"]
        text = context["complaint"]
        email = context.get("customer_email") or (EMAIL_RE.search(text).group(0) if EMAIL_RE.search(text) else None)
        n = len([m for m in messages if m["role"] == "assistant"]) + 1

        def tool_turn(*calls):
            return {"role": "assistant", "content": None, "tool_calls": list(calls)}

        def final(msg):
            return {"role": "assistant", "content": msg}

        ticket = results.get("register_complaint", {}).get("ticket_id", "your ticket")

        # 1) first turn: register ticket + look up the customer
        if not results:
            calls = [_call(1, "register_complaint", customer_email=email, category=triage["category"],
                           urgency=triage["urgency"], summary=text[:300])]
            if email:
                calls.append(_call(2, "get_customer", email=email))
            return tool_turn(*calls)

        # 2) terminal outcomes
        if "issue_refund" in results:
            r = results["issue_refund"]
            if r.get("ok"):
                return final(f"I've issued a refund of ${r['amount']:.2f} for order {r['order_id']} "
                             f"(refund {r['refund_id']}). It will reach your original payment method in 5-7 business days. "
                             f"Your ticket is {ticket}. Sorry for the trouble.")
            if r.get("error") == "rejected_by_human":
                note = r.get("note") or "it did not meet our refund criteria"
                return final(f"Thanks for your patience. A support lead reviewed your refund request and could not approve it: {note}. "
                             f"Your ticket {ticket} is closed as 'refund_rejected'; reply to this ticket if you'd like it reviewed again.")
            return self._escalate_or_final(results, n, f"refund could not be processed ({r.get('detail')})", ticket, final, tool_turn)
        if "escalate_to_human" in results:
            r = results["escalate_to_human"]
            return final(f"I've escalated your case to our human support team ({r.get('reason', '')}). "
                         f"Your ticket is {ticket}; a specialist will contact you shortly.")

        # 3) no email supplied
        if not email:
            return final(f"I've logged ticket {ticket}, but I couldn't identify your account. "
                         "Please reply with the email address on your account so we can look into this.")

        # 4) customer lookup result
        cust = results.get("get_customer", {})
        if not cust.get("ok"):
            return self._escalate_or_final(results, n, f"customer {email} not found in our records", ticket, final, tool_turn)
        customer = cust["customer"]

        cat = triage["category"]
        if cat == "account":
            if triage["urgency"] == "high":
                return tool_turn(_call(n, "escalate_to_human", reason="possible account security incident"))
            return final(f"Hi {customer['name']}, you can reset your password from the sign-in page using 'Forgot password' "
                         f"(per our Account Help policy). You can update your details under Account Settings. Ticket: {ticket}.")

        # 5) pick order, fetch shipping status
        order_id = ORDER_RE.search(text)
        order_id = order_id.group(0).upper() if order_id else (customer["orders"][0]["id"] if customer["orders"] else None)
        if not order_id:
            return self._escalate_or_final(results, n, "no order found on the account", ticket, final, tool_turn)
        if "get_order_status" not in results:
            return tool_turn(_call(n, "get_order_status", order_id=order_id))
        st = results["get_order_status"]
        if not st.get("ok"):
            return self._escalate_or_final(results, n, f"shipping service unavailable ({st.get('detail')})", ticket, final, tool_turn)
        ship = st["shipment"]
        asked = MONEY_RE.search(text)
        amount = min(float(asked.group(1)), ship["total"]) if asked else ship["total"]
        low = text.lower()

        if cat == "product_defect":
            d = ship["days_since_delivery"]
            if d is not None and d <= 30:
                return tool_turn(_call(n, "issue_refund", order_id=order_id, amount=amount,
                                       reason=f"defective item, {d} days after delivery (refund policy: 30-day window)"))
            return tool_turn(_call(n, "escalate_to_human",
                                   reason="defect outside 30-day refund window; warranty claim needs a specialist"))
        if cat == "delivery":
            if ship["status"] == "delivered":
                return tool_turn(_call(n, "escalate_to_human", reason="marked delivered but customer disputes receipt; carrier investigation"))
            if ship["days_late"] > 7:
                return tool_turn(_call(n, "issue_refund", order_id=order_id, amount=ship["total"],
                                       reason=f"order {ship['days_late']} days past estimated delivery (shipping policy: >7 days)"))
            return final(f"Your order {order_id} ({ship['item']}) is {ship['status'].replace('_', ' ')} with {ship['carrier']} "
                         f"(tracking {ship['tracking']}), expected {ship['expected_delivery']}. It is within the normal delivery window, "
                         f"so no refund applies yet. Ticket: {ticket}.")
        # billing
        if any(k in low for k in ("twice", "double", "duplicate", "refund", "overcharg")):
            return tool_turn(_call(n, "issue_refund", order_id=order_id, amount=amount,
                                   reason="billing correction: duplicate/incorrect charge (refund policy)"))
        return final(f"I've reviewed order {order_id} (${ship['total']:.2f}) and don't see an incorrect charge. "
                     f"A billing specialist can look further if you reply with details. Ticket: {ticket}.")

    @staticmethod
    def _escalate_or_final(results, n, reason, ticket, final, tool_turn):
        if "escalate_to_human" in results:
            return final(f"We hit a problem handling this automatically ({reason}). Your ticket {ticket} has been "
                         "passed to our human support team and they'll follow up.")
        return tool_turn(_call(n, "escalate_to_human", reason=reason[:290]))


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a customer-support resolution agent for an online store.
Use the tools to resolve the complaint. Always call register_complaint first, then get_customer.
Before any refund, call get_order_status to verify the order total, status and delivery date against the policy
(e.g. the 30-day defect window). Ground decisions ONLY in the policy excerpts provided.
Issue refunds with issue_refund only when policy allows; refunds above the approval limit are routed to a human
automatically, so still call issue_refund and wait for the result. A successful issue_refund result means the refund
is done (if it says approved_by_human, a support lead already signed it off). If the customer is unknown, the issue is
outside policy, or a tool keeps failing, call escalate_to_human. Finish with a short, polite plain-text message
(no Markdown) to the customer that states the outcome, the ticket id and any refund id.
Never invent order data."""


class OpenAILLM:
    def __init__(self, settings: Settings):
        from openai import OpenAI

        self.s = settings
        self.name = f"openai-compatible:{settings.model}"
        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.base_url or None,
                             timeout=settings.llm_timeout_s, max_retries=0)

    def triage(self, text: str) -> Triage:
        prompt = ('Classify this customer complaint. Reply with ONLY JSON: {"category": one of '
                  '"billing"|"delivery"|"product_defect"|"account", "urgency": "low"|"medium"|"high", "reason": short string}.\n\n'
                  f"Complaint: {text}")
        try:
            r = self.client.chat.completions.create(model=self.s.model, temperature=self.s.llm_temperature,
                                                    messages=[{"role": "user", "content": prompt}])
            raw = r.choices[0].message.content or ""
            m = re.search(r"\{.*\}", raw, re.S)  # tolerates ```json fences and prose around the JSON
            return Triage(**json.loads(m.group(0)))
        except (ValidationError, ValueError, AttributeError) as exc:
            raise LLMError(f"unparseable triage output: {exc}") from exc
        except Exception as exc:
            raise LLMError(str(exc)) from exc

    @property
    def _is_gemini(self) -> bool:
        return "generativelanguage.googleapis.com" in (self.s.base_url or "") or self.s.model.startswith("gemini")

    def _outbound(self, messages: list[dict]) -> list[dict]:
        """Make stored history acceptable to strict OpenAI-compatible backends.

        Gemini 3.x thinking models return a `thought_signature` in `tool_calls[i].extra_content` and reject
        the next request (HTTP 400) if a function call in the history comes back without it. We store it
        verbatim (see chat()), and for tool calls the system injected itself (e.g. the fallback escalation)
        we add Google's documented dummy signature so validation is skipped for that one call.
        """
        out = []
        for m in messages:
            m = dict(m)
            if m.get("role") == "assistant" and m.get("tool_calls"):
                calls = [dict(tc) for tc in m["tool_calls"]]
                if self._is_gemini and not any(tc.get("extra_content") for tc in calls):
                    calls[0]["extra_content"] = {"google": {"thought_signature": "skip_thought_signature_validator"}}
                if not self._is_gemini:
                    for tc in calls:
                        tc.pop("extra_content", None)
                m["tool_calls"] = calls
                if m.get("content") is None:
                    m["content"] = ""
            out.append(m)
        return out

    def chat(self, messages: list[dict], tools: list[dict], context: dict) -> dict:
        system = SYSTEM_PROMPT + "\n\nTriage: " + json.dumps(context["triage"]) + "\n\nPolicy excerpts:\n" + context["policy_text"]
        if context.get("customer_email"):
            system += f"\n\nCustomer email (from session): {context['customer_email']}"
        try:
            r = self.client.chat.completions.create(
                model=self.s.model, temperature=self.s.llm_temperature, tools=tools,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": context["complaint"]}, *self._outbound(messages)])
            m = r.choices[0].message
        except Exception as exc:
            raise LLMError(str(exc)) from exc
        out: dict = {"role": "assistant", "content": m.content}
        if m.tool_calls:
            calls = []
            for i, t in enumerate(m.tool_calls):
                d = {"id": t.id or f"call_{i}_{t.function.name}", "type": "function",
                     "function": {"name": t.function.name, "arguments": t.function.arguments or "{}"}}
                extra = (getattr(t, "model_extra", None) or {}).get("extra_content") or getattr(t, "extra_content", None)
                if isinstance(extra, dict):
                    d["extra_content"] = extra  # Gemini thought signature: must be sent back unchanged
                calls.append(d)
            out["tool_calls"] = calls
        elif not (m.content or "").strip():
            raise LLMError("empty completion (no text and no tool calls)")
        return out


def build_llm(settings: Settings) -> LLM:
    if settings.effective_mode == "openai":
        return OpenAILLM(settings)  # type: ignore[return-value]
    return MockLLM(settings.mock_delay_s)
