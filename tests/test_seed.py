from app import db
from app.config import Settings
from app.service import SupportService


def test_seed_off_by_default(svc):
    assert db.list_tickets(svc.settings.db_path) == []


def test_env_default_follows_llm_mode(monkeypatch):
    monkeypatch.delenv("DEMO_SEED", raising=False)
    monkeypatch.setenv("LLM_MODE", "mock")
    assert Settings.from_env().demo_seed is True
    monkeypatch.setenv("LLM_MODE", "auto")
    assert Settings.from_env().demo_seed is False
    monkeypatch.setenv("DEMO_SEED", "1")
    assert Settings.from_env().demo_seed is True


def test_seed_populates_once_and_pending_is_approvable(tmp_path):
    st = Settings(llm_mode="mock", data_dir=tmp_path, backoff_base_s=0.0, tool_timeout_s=2.0, demo_seed=True)
    svc = SupportService(st)
    tk = db.list_tickets(st.db_path)
    assert len(tk) == 6 and all(t["summary"].startswith("[Sample]") for t in tk)
    assert sorted(t["status"] for t in tk) == ["escalated", "pending_approval", "resolved", "resolved", "resolved", "resolved"]
    assert sorted(r["amount"] for r in db.list_refunds(st.db_path)) == [24.99, 259.0]
    pend = svc.pending_approvals()
    assert len(pend) == 1
    # restart with data present: no duplicates
    svc2 = SupportService(st)
    assert len(db.list_tickets(st.db_path)) == 6
    assert svc2.decide(pend[0]["thread_id"], True)["outcome"] == "refunded"
    assert svc2.pending_approvals() == []
