"""Run the three README scenarios against the configured REAL model and print a short transcript.

Example (Gemini, key mapped at runtime, never written to disk):
    API_KEY_ENV=GEMINI_API_KEY BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
    MODEL=gemini-2.5-flash LLM_MODE=openai DATA_DIR=/tmp/sa-real .venv/bin/python scripts/real_run.py
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import Settings  # noqa: E402
from app.service import SupportService  # noqa: E402

SCENARIOS = [
    ("1 small refund", "alice@example.com", "The USB-C cable from ORD-1001 is defective. Please refund $24.99."),
    ("2 large refund", "alice@example.com", "My headphones from ORD-1002 stopped working after a week. I want a full refund."),
    ("3 unknown customer", "ghost@nowhere.com", "I was charged twice for my last order. Please fix this."),
]


def show(v):
    for s in v["steps"]:
        d = s.get("detail", {})
        extra = ""
        if s["node"] == "tools":
            extra = f"  args={json.dumps(d.get('args'))[:110]}"
        print(f"   [{s['node']:8}] {s['title']}{extra}")


def main():
    s = Settings.from_env()
    if s.effective_mode != "openai":
        sys.exit("Set an API key (OPENAI_API_KEY or API_KEY_ENV) to run against a real model.")
    shutil.rmtree(s.data_dir, ignore_errors=True)
    svc = SupportService(s)
    print(f"LLM: {svc.llm.name}  base_url: {s.base_url}  retrieval: {svc.kb.backend}\n")
    ok = True
    for name, email, msg in SCENARIOS:
        t0 = time.time()
        print(f"== Scenario {name}: {email}: {msg}")
        v = svc.start(msg, email, wait=True)
        if v["status"] == "awaiting_approval":
            show(v)
            p = v["pending_approval"]
            print(f"   >> PAUSED for approval: {p['action']} {p['order_id']} ${p['amount']} ({p['why_approval_needed']})")
            n = len(v["steps"])
            v = svc.decide(v["thread_id"], True, "Verified defect", "support-lead", wait=True)
            v = {**v, "steps": v["steps"][n:]}
            print("   >> human APPROVED, resumed:")
        show(v)
        t = v.get("ticket") or {}
        print(f"   status={v['status']} outcome={v['outcome']} ticket={t.get('id')}:{t.get('status')}  ({time.time()-t0:.1f}s)")
        print(f"   reply: {v['reply']}\n")
        ok &= v["status"] == "completed"
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
