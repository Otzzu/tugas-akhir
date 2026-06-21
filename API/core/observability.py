"""Production hardening — logging setup, request middleware, error handler."""
from __future__ import annotations

import logging
import time

from fastapi import Request
from fastapi.responses import JSONResponse

log = logging.getLogger("api")

_configured = False


def setup_logging() -> None:
    """Configure stdlib logging once."""
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _configured = True


async def logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000
        log.error("%s %s failed after %.1fms", request.method, request.url.path,
                  latency_ms, exc_info=True)
        raise
    latency_ms = (time.perf_counter() - start) * 1000
    log.info("%s %s %d %.1fms", request.method, request.url.path,
             response.status_code, latency_ms)
    return response


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled exception on %s %s", request.method, request.url.path,
              exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error", "type": type(exc).__name__},
    )
