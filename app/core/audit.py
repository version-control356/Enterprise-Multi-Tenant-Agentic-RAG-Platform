"""Structured audit logging for security-sensitive tenant actions."""

import json
import logging
from typing import Any, Optional

from fastapi import Request


audit_logger = logging.getLogger("audit")


def record_audit_event(
    event: str,
    request: Request,
    *,
    outcome: str,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Write an audit event without recording passwords, tokens, or prompts."""
    payload = {
        "event": event,
        "outcome": outcome,
        "request_id": getattr(request.state, "request_id", None),
        "client_ip": request.client.host if request.client else None,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "metadata": metadata or {},
    }
    audit_logger.info(json.dumps(payload, sort_keys=True, separators=(",", ":")))
