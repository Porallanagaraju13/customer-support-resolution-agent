import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.service import SupportService


@pytest.fixture
def settings(tmp_path):
    return Settings(llm_mode="mock", data_dir=tmp_path, backoff_base_s=0.0, tool_timeout_s=2.0)


@pytest.fixture
def svc(settings):
    return SupportService(settings)


@pytest.fixture
def client(svc):
    return TestClient(create_app(service=svc))
