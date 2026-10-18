"""Runtime configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"
    base_url: str = ""
    llm_mode: str = "auto"  # auto | mock | openai
    embeddings: str = "local"  # local (TF-IDF) | openai
    embedding_model: str = "text-embedding-3-small"
    data_dir: Path = ROOT / "data"
    refund_threshold: float = 50.0
    tool_timeout_s: float = 5.0
    tool_max_retries: int = 3
    backoff_base_s: float = 0.2
    mock_delay_s: float = 0.0
    llm_timeout_s: float = 30.0
    llm_temperature: float = 0.0
    demo_seed: bool = False  # seed sample tickets into an empty DB (see app/seed.py)

    @property
    def db_path(self) -> str:
        return str(Path(self.data_dir) / "support.db")

    @property
    def checkpoint_path(self) -> str:
        return str(Path(self.data_dir) / "checkpoints.db")

    @property
    def effective_mode(self) -> str:
        if self.llm_mode == "mock":
            return "mock"
        if self.llm_mode == "openai":
            return "openai"
        return "openai" if self.openai_api_key else "mock"

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        seed_default = "1" if e("LLM_MODE", "auto").lower() == "mock" else "0"
        return cls(
            demo_seed=e("DEMO_SEED", seed_default).strip().lower() in ("1", "true", "yes", "on"),
            openai_api_key=e("OPENAI_API_KEY", "") or (e(e("API_KEY_ENV", ""), "") if e("API_KEY_ENV") else ""),
            model=e("MODEL", "gpt-4o-mini"),
            base_url=e("BASE_URL", ""),
            llm_mode=e("LLM_MODE", "auto").lower(),
            embeddings=e("EMBEDDINGS", "local").lower(),
            embedding_model=e("EMBEDDING_MODEL", "text-embedding-3-small"),
            data_dir=Path(e("DATA_DIR", str(ROOT / "data"))),
            refund_threshold=float(e("REFUND_APPROVAL_THRESHOLD", "50")),
            tool_timeout_s=float(e("TOOL_TIMEOUT_S", "5")),
            tool_max_retries=int(e("TOOL_MAX_RETRIES", "3")),
            backoff_base_s=float(e("TOOL_BACKOFF_BASE_S", "0.2")),
            mock_delay_s=float(e("MOCK_DELAY_S", "0")),
            llm_timeout_s=float(e("LLM_TIMEOUT_S", "30")),
            llm_temperature=float(e("LLM_TEMPERATURE", "0")),
        )
