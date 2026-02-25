"""Observability and distributed tracing module with native Langfuse integration."""

import time
import logging
import asyncio
import httpx
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional
from pydantic import BaseModel, Field
from app.config import settings

logger = logging.getLogger("telemetry")

MAX_TRACE_HISTORY = 500


class SpanRecord(BaseModel):
    """Execution span capturing timing and metadata for a discrete sub-operation."""

    span_id: str
    name: str
    start_time: float
    end_time: float
    duration_ms: float
    status: str = "ok"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraceRecord(BaseModel):
    """Complete trace record capturing end-to-end request lifecycle and spans."""

    trace_id: str
    tenant_id: str
    user_id: str
    role: str
    query_preview: str
    created_at: float
    total_duration_ms: float
    cache_hit: bool = False
    rewrite_count: int = 0
    chunks_retrieved: int = 0
    spans: List[SpanRecord] = Field(default_factory=list)
    status: str = "ok"
    error_message: Optional[str] = None


class TelemetryTracker:
    """In-memory circular telemetry collector with optional Langfuse cloud/on-prem sync."""

    def __init__(self, maxlen: int = MAX_TRACE_HISTORY):
        self._traces: deque[TraceRecord] = deque(maxlen=maxlen)
        self._active_spans: Dict[str, List[SpanRecord]] = {}

    def start_trace(self, trace_id: str) -> None:
        """Initialize active spans list for a trace."""
        self._active_spans[trace_id] = []

    @asynccontextmanager
    async def record_span(
        self,
        trace_id: str,
        span_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Asynchronously measure and record an execution span."""
        start = time.perf_counter()
        span_meta = metadata.copy() if metadata else {}
        status = "ok"
        try:
            yield span_meta
        except Exception as exc:
            status = "error"
            span_meta["error"] = str(exc)
            raise
        finally:
            end = time.perf_counter()
            duration_ms = round((end - start) * 1000, 2)
            span = SpanRecord(
                span_id=f"{span_name}_{int(start * 1000)}",
                name=span_name,
                start_time=round(start, 4),
                end_time=round(end, 4),
                duration_ms=duration_ms,
                status=status,
                metadata=span_meta,
            )
            if trace_id in self._active_spans:
                self._active_spans[trace_id].append(span)

    def finalize_trace(
        self,
        trace_id: str,
        tenant_id: str,
        user_id: str,
        role: str,
        query: str,
        start_time: float,
        cache_hit: bool = False,
        rewrite_count: int = 0,
        chunks_retrieved: int = 0,
        error_message: Optional[str] = None,
    ) -> TraceRecord:
        """Assemble, store, and asynchronously sync trace records with Langfuse."""
        total_duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        spans = self._active_spans.pop(trace_id, [])

        query_preview = (query[:80] + "...") if len(query) > 80 else query

        record = TraceRecord(
            trace_id=trace_id,
            tenant_id=tenant_id,
            user_id=user_id,
            role=role,
            query_preview=query_preview,
            created_at=round(start_time, 2),
            total_duration_ms=total_duration_ms,
            cache_hit=cache_hit,
            rewrite_count=rewrite_count,
            chunks_retrieved=chunks_retrieved,
            spans=spans,
            status="error" if error_message else "ok",
            error_message=error_message,
        )
        self._traces.appendleft(record)

        if settings.langfuse_configured:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._export_to_langfuse(record))
            except RuntimeError:
                pass

        return record

    async def _export_to_langfuse(self, record: TraceRecord) -> None:
        """Forward trace records to Langfuse Cloud or Self-hosted endpoint via REST API."""
        try:
            url = f"{settings.LANGFUSE_HOST.rstrip('/')}/api/public/ingestion"
            events = [
                {
                    "id": f"trace_{record.trace_id}",
                    "type": "trace-create",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created_at)),
                    "body": {
                        "id": record.trace_id,
                        "name": "AgenticRAG_Query",
                        "userId": record.user_id,
                        "metadata": {
                            "tenant_id": record.tenant_id,
                            "role": record.role,
                            "cache_hit": record.cache_hit,
                            "rewrite_count": record.rewrite_count,
                            "duration_ms": record.total_duration_ms,
                        },
                        "input": {"query": record.query_preview},
                        "tags": [record.tenant_id, record.role],
                    },
                }
            ]

            for span in record.spans:
                events.append({
                    "id": f"span_{span.span_id}",
                    "type": "span-create",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(span.start_time)),
                    "body": {
                        "id": span.span_id,
                        "traceId": record.trace_id,
                        "name": span.name,
                        "startTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(span.start_time)),
                        "endTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(span.end_time)),
                        "metadata": span.metadata,
                        "statusMessage": span.status,
                    }
                })

            async with httpx.AsyncClient(timeout=4.0) as client:
                auth = (settings.LANGFUSE_PUBLIC_KEY, settings.LANGFUSE_SECRET_KEY)
                resp = await client.post(url, json={"batch": events}, auth=auth)
                if not resp.is_success:
                    logger.warning(f"Langfuse ingestion returned {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"Failed to export trace {record.trace_id} to Langfuse: {e}")

    def get_traces_for_tenant(
        self,
        tenant_id: str,
        limit: int = 50
    ) -> List[TraceRecord]:
        """Retrieve recent traces filtered by tenant."""
        return [t for t in self._traces if t.tenant_id == tenant_id][:limit]

    def get_metrics_summary(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Compute aggregated performance metrics."""
        traces = [t for t in self._traces if tenant_id is None or t.tenant_id == tenant_id]
        if not traces:
            return {
                "total_requests": 0,
                "cache_hit_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "error_rate": 0.0,
                "langfuse_enabled": settings.langfuse_configured,
            }

        total = len(traces)
        cache_hits = sum(1 for t in traces if t.cache_hit)
        errors = sum(1 for t in traces if t.status == "error")
        latencies = sorted(t.total_duration_ms for t in traces)

        p95_idx = int(total * 0.95)
        p95_latency = latencies[min(p95_idx, total - 1)]

        return {
            "total_requests": total,
            "cache_hit_rate": round((cache_hits / total) * 100, 2),
            "avg_latency_ms": round(sum(latencies) / total, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "error_rate": round((errors / total) * 100, 2),
            "langfuse_enabled": settings.langfuse_configured,
        }


    def clear_traces(self, tenant_id: Optional[str] = None) -> int:
        """Purge recorded trace history for a tenant or globally."""
        if tenant_id is None:
            count = len(self._traces)
            self._traces.clear()
            self._active_spans.clear()
            return count

        remaining = deque(
            (t for t in self._traces if t.tenant_id != tenant_id),
            maxlen=self._traces.maxlen,
        )
        removed_count = len(self._traces) - len(remaining)
        self._traces = remaining
        return removed_count


telemetry_tracker = TelemetryTracker()
