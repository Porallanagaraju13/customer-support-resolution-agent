def test_health_config_and_index(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    cfg = client.get("/api/config").json()
    assert cfg["llm_mode"] == "mock" and cfg["refund_approval_threshold"] == 50
    assert "Customer Support Resolution Agent" in client.get("/").text


def test_full_api_flow(client):
    r = client.post("/api/chat", json={"message": "My headphones from ORD-1002 stopped working, full refund please",
                                       "customer_email": "alice@example.com"}).json()
    assert r["status"] == "awaiting_approval"
    tid = r["thread_id"]
    assert client.get("/api/approvals").json()[0]["thread_id"] == tid
    assert client.get("/api/tickets").json()[0]["status"] == "pending_approval"

    done = client.post(f"/api/approvals/{tid}", json={"decision": "approve", "note": "ok"}).json()
    assert done["status"] == "completed" and done["outcome"] == "refunded"
    assert client.post(f"/api/approvals/{tid}", json={"decision": "approve"}).status_code == 409
    assert client.get(f"/api/threads/{tid}").json()["status"] == "completed"
    assert client.get("/api/tickets/TKT-0001").json()["status"] == "resolved"
    assert len(client.get("/api/refunds").json()) == 1


def test_validation_and_404s(client):
    assert client.post("/api/chat", json={"message": "x"}).status_code == 422
    assert client.get("/api/threads/nope").status_code == 404
    assert client.post("/api/approvals/nope", json={"decision": "approve"}).status_code == 404
    assert client.post("/api/approvals/nope", json={"decision": "maybe"}).status_code == 422
    assert client.get("/api/tickets/TKT-9999").status_code == 404


def test_async_chat_can_be_polled(client):
    r = client.post("/api/chat", json={"message": "Where is my keyboard ORD-1003 delivery?", "customer_email": "bob@example.com", "wait": False}).json()
    import time
    for _ in range(50):
        v = client.get(f"/api/threads/{r['thread_id']}").json()
        if v["status"] == "completed":
            break
        time.sleep(0.1)
    assert v["status"] == "completed"
