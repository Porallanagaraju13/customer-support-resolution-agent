"""OpenAI-compatible provider is exercised against a fake client (no network, no key)."""
from types import SimpleNamespace as NS

from app.config import Settings
from app.llm import OpenAILLM, build_llm, MockLLM
from app.tools import openai_tool_schemas


def fake_client(content=None, tool_calls=None):
    msg = NS(content=content, tool_calls=tool_calls)
    create = lambda **kw: NS(choices=[NS(message=msg)])
    return NS(chat=NS(completions=NS(create=create)))


def test_mode_selection():
    assert isinstance(build_llm(Settings()), MockLLM)
    assert isinstance(build_llm(Settings(llm_mode="mock", openai_api_key="k")), MockLLM)
    assert Settings(openai_api_key="k").effective_mode == "openai"


def test_openai_triage_and_tool_calls():
    llm = OpenAILLM(Settings(openai_api_key="test", base_url="http://localhost:9"))
    llm.client = fake_client('Sure: {"category":"billing","urgency":"high","reason":"x"}')
    assert llm.triage("charged twice").category == "billing"
    tc = NS(id="t1", function=NS(name="get_customer", arguments='{"email":"a@b.co"}'))
    llm.client = fake_client(None, [tc])
    ctx = {"triage": {}, "complaint": "hi", "policy_text": "", "customer_email": None}
    out = llm.chat([], openai_tool_schemas(), ctx)
    assert out["tool_calls"][0]["function"]["name"] == "get_customer"


def test_gemini_thought_signature_round_trip():
    """Gemini 3.x returns tool_calls[].extra_content.google.thought_signature and 400s if it is not sent back."""
    s = Settings(openai_api_key="test", base_url="https://generativelanguage.googleapis.com/v1beta/openai/", model="gemini-3-flash")
    llm = OpenAILLM(s)
    sent = {}

    def create(**kw):
        sent.update(kw)
        tc = NS(id="call_1", function=NS(name="get_customer", arguments='{"email":"a@b.co"}'),
                model_extra={"extra_content": {"google": {"thought_signature": "SIG"}}})
        return NS(choices=[NS(message=NS(content=None, tool_calls=[tc]))])

    llm.client = NS(chat=NS(completions=NS(create=create)))
    ctx = {"triage": {}, "complaint": "hi", "policy_text": "", "customer_email": None}
    out = llm.chat([], openai_tool_schemas(), ctx)
    assert out["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "SIG"
    # history goes back with the signature intact; a system-injected call (no signature) gets the documented dummy
    injected = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_fallback_escalate", "type": "function", "function": {"name": "escalate_to_human", "arguments": "{}"}}]}
    llm.chat([out, {"role": "tool", "tool_call_id": "call_1", "name": "get_customer", "content": "{}"}, injected],
             openai_tool_schemas(), ctx)
    hist = sent["messages"]
    assert hist[2]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "SIG"
    assert hist[4]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "skip_thought_signature_validator"
    assert hist[4]["content"] == ""


def test_non_gemini_strips_extra_content():
    llm = OpenAILLM(Settings(openai_api_key="test", base_url="http://localhost:9", model="gpt-4o-mini"))
    msg = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "x", "arguments": "{}"}, "extra_content": {"google": {}}}]}
    assert "extra_content" not in llm._outbound([msg])[0]["tool_calls"][0]


def test_api_key_env_mapping(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("API_KEY_ENV", "MY_PROVIDER_KEY")
    monkeypatch.setenv("MY_PROVIDER_KEY", "k-123")
    assert Settings.from_env().openai_api_key == "k-123"
    assert Settings.from_env().effective_mode == "openai"
