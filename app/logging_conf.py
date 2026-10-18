"""Structured (JSON-lines) logging."""
from __future__ import annotations

import json
import logging
import sys
import time

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if any(getattr(h, "_support_agent", False) for h in root.handlers):
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(JsonFormatter())
    h._support_agent = True  # type: ignore[attr-defined]
    root.addHandler(h)
    root.setLevel(level)


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    logging.getLogger("support_agent").log(level, event, extra=fields)
