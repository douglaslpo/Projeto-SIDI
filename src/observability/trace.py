"""Observabilidade minima: structured logs com trace_id por requisicao."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger("portfolio")
logger.setLevel(logging.INFO)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)


def log_event(event: str, **fields: Any) -> None:
    """Emite 1 linha de log estruturado JSON."""
    trace_id = fields.pop("trace_id", None) or str(uuid.uuid4())
    payload = {
        "ts": time.time(),
        "event": event,
        "trace_id": trace_id,
        **fields,
    }
    logger.info(json.dumps(payload, default=str))


@contextmanager
def trace(operation: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Context manager que mede latencia e emite log de inicio + fim."""
    tid = fields.pop("trace_id", None) or str(uuid.uuid4())
    start = time.perf_counter()
    log_event(f"{operation}_start", trace_id=tid, **fields)
    ctx: dict[str, Any] = {"trace_id": tid}
    try:
        yield ctx
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        log_event(f"{operation}_end", trace_id=tid, latency_ms=latency_ms, **fields)
