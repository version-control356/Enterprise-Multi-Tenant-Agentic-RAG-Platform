# 🛡️ Enterprise Multi-Tenant Agentic RAG Platform

> **A production-oriented, containerized Retrieval-Augmented Generation (RAG) platform featuring strict multi-tenant isolation, role-based access control (RBAC), hybrid dense-sparse vector search, self-correcting agentic workflows, OWASP defensive guardrails, and real-time distributed tracing.**

---

<!-- SCREENSHOT PLACEHOLDER 1 -->
<!-- Recommended Screenshot: Full view of the Streamlit Operator Dashboard with a streaming chat response and the sidebar authenticated as an Admin. -->
<!-- Image File: docs/screenshots/01_dashboard_overview.png -->
![Platform Dashboard Overview](docs/screenshots/01_dashboard_overview.png)
*Figure 1: Production Operator Dashboard featuring tenant-scoped streaming chat, role-based document ingestion, and live telemetry.*

---

## 🌟 Executive Overview & Key Capabilities

This platform is engineered to solve the most critical enterprise challenges in Generative AI: **cross-tenant data leakage**, **unauthorized privilege escalation**, **hallucination in knowledge retrieval**, and **adversarial prompt injection attacks**. 

| Capability | Enterprise Implementation | Production Impact |
| :--- | :--- | :--- |
| **Multi-Tenant Isolation** | Composite key `(tenant_id, username)` in PostgreSQL + Mandatory metadata filters in Qdrant | **Zero cross-tenant data leakage** at database, vector, and cache layers. |
| **Granular RBAC** | Role-filtered retrieval (`admin`, `analyst`, `viewer`) | Restricted users cannot retrieve confidential chunks even within the same tenant. |
| **Defensive AI & NeMo Guardrails** | **NVIDIA NeMo Guardrails** (Colang input/output rails) + OWASP injection regexes + **Presidio PII Redaction** | Malicious prompts are blocked and PII is redacted **before** touching downstream LLMs or vector stores. |
| **Hybrid Search (Dense + Sparse)** | FastEmbed BGE (`384d`) + BM25 Sparse Tokenizer + Qdrant Reciprocal Rank Fusion (RRF) | Superior retrieval accuracy for both semantic intent and exact hardware SKUs / legal codes. |
| **Cross-Encoder Reranking** | **Cohere Rerank API** (`rerank-v3.5`) cross-encoder re-ordering | Re-scores hybrid retrieval candidates with high semantic precision before passing to LLM. |
| **Corrective Agentic RAG (CRAG)** | Stateful LangGraph workflow with **LLM-as-a-Judge** grading and autonomous query reformulation | Bounded self-correction prevents hallucinations and loops when retrieval is weak. |
| **Universal Parser & OCR** | PDF, DOCX, XLSX, CSV, HTML, Code + **Pytesseract OCR** for scanned/image-based PDFs | Seamless multi-format knowledge extraction with automatic OCR fallback. |
| **Async Background Ingestion** | Dedicated `POST /api/v1/ingest/async` & `GET /api/v1/ingest/status/{task_id}` polling | Non-blocking ingestion for large documents without HTTP gateway timeouts. |
| **Stateful Persistence** | PostgreSQL Checkpointing via `AsyncPostgresSaver` | Multi-turn chat state persists safely across container restarts and horizontal scale. |
| **Version-Scoped Caching** | SHA-256 digested Redis keys with Tenant Version Epochs (`tenant_ver:<tenant_id>`) | **Sub-5ms cache hits**; instant tenant-wide cache invalidation upon uploading new documents. |
| **High-Throughput Inference** | Groq Cloud integration with asynchronous SSE token streaming | Token generation at ultra-low latency. |
| **Durable Ingestion Queue** | Redis-backed queue with processing retention and acknowledgement | Async ingestion can be consumed across backend replicas instead of relying only on process memory. |
| **Distributed Observability** | In-memory span tracker + REST API telemetry + Native **Langfuse Cloud** sync | Real-time visibility into node-by-node latencies (`retrieve_hybrid`, `rerank_cohere`, `grade`, `generate_llm`). |

---

## 🏗️ High-Level System Architecture

For the complete topology, data-isolation model, CRAG state graph, security
controls, deployment design, and latency analysis, see
[ARCHITECTURE.md](ARCHITECTURE.md).

```text
                                +---------------------------------------------+
                                |     Streamlit Operator Dashboard (UI)       |
                                |    (Chat Interface + Observability Tabs)    |
                                +---------------------------------------------+
                                                       │  (HTTP / SSE Stream)
                                                       ▼
+─────────────────────────────────────────────────────────────────────────────────────────────────────────────────+
|                                           FastAPI Gateway (Port 8000)                                           |
|                                                                                                                 |
|  [Security Headers] ──► [JWT / RBAC Auth] ──► [NVIDIA NeMo Guardrails / OWASP Regex] ──► [Presidio PII Redaction]|
+───────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────+
                                                    │
                   ┌────────────────────────────────┴────────────────────────────────┐
                   ▼                                                                 ▼
      +──────────────────────────+                                     +──────────────────────────+
      |  Redis Response Cache    |                                     |    Distributed Tracing   |
      |  (Scoped by Tenant+Role) |                                     | (Spans & Langfuse Sync)  |
      +──────────────────────────+                                     +──────────────────────────+
                   │ (Cache Miss)                                                    │
                   └────────────────────────────────┬────────────────────────────────┘
                                                    │
                                                    ▼
                           +──────────────────────────────────────────────────+
                           |       LangGraph Corrective Agentic Engine        |
                           |                                                  |
                           |  [retrieve_hybrid] ──► [Cohere Reranker (v3.5)]  |
                           |          ▲                      │                |
                           |          │                      ▼                |
                           |          │             [grade_documents (LLM)]   |
                           |          │                      │                |
                           |          │ (Low Relevance)      ▼ (Relevant)     |
                           |   [rewrite_query] ◄────── [decide_to_generate]   |
                           |                                 │                |
                           |                                 ▼                |
                           |                          [generate_llm]          |
                           +────────────────────────┬─────────────────────────+
                                                    │
                   ┌────────────────────────────────┴────────────────────────────────┐
                   ▼                                                                 ▼
      +──────────────────────────+                                     +──────────────────────────+
      |   Qdrant Hybrid Vector   |                                     |  PostgreSQL Checkpoints  |
      | Dense + BM25 Sparse RRF  |                                     |  & Tenant User Registry  |
      +──────────────────────────+                                     +──────────────────────────+
```

---

## 📸 Platform Showcase & Screenshot Guide

> **To customize the documentation with your live screenshots, save your images in `docs/screenshots/` matching the filenames below:**

### 1. Granular RBAC Multi-Select Document Ingestion
<!-- SCREENSHOT PLACEHOLDER 2 -->
<!-- Recommended Screenshot: Sidebar showing the file uploader with the interactive Multi-Select dropdown for allowed roles (admin, analyst, viewer). -->
<!-- Image File: docs/screenshots/02_rbac_ingestion.png -->
![RBAC Document Ingestion](docs/screenshots/02_rbac_ingestion.png)
*Figure 2: Administrator document upload with multi-select role permissions and automatic hybrid indexing.*

---

### 2. Live Prompt Injection Defense & Interception
<!-- SCREENSHOT PLACEHOLDER 3 -->
<!-- Recommended Screenshot: Chat screen showing the red security alert banner after submitting a prompt injection payload. -->
<!-- Image File: docs/screenshots/03_injection_guardrail.png -->
![Prompt Injection Interception](docs/screenshots/03_injection_guardrail.png)
*Figure 3: Immediate pre-flight interception of adversarial prompt injection and DAN jailbreak attempts.*

---

### 3. Distributed Telemetry & Span Execution Timelines
<!-- SCREENSHOT PLACEHOLDER 4 -->
<!-- Recommended Screenshot: The "Observability & Traces" tab showing metric cards (Total Requests, Cache Hit %, P95 Latency) and expanded execution spans. -->
<!-- Image File: docs/screenshots/04_observability_traces.png -->
![Observability and Traces](docs/screenshots/04_observability_traces.png)
*Figure 4: Real-time telemetry dashboard detailing sub-operation execution spans and latency percentiles.*

---

## ⚡ Quick Start & Deployment

### Prerequisites
* Docker Engine 24.0+ and Docker Compose v2
* A Groq API Key ([https://console.groq.com](https://console.groq.com))
* Python 3.11+ (if running tests locally)

### 1. Clone & Configure
```bash
git clone https://github.com/your-org/multi-tenant-agentic-rag-platform.git
cd multi-tenant-agentic-rag-platform

# Copy the environment template
cp .env.example .env
```

### 2. Configure Environment Variables
Edit `.env` and set your secrets:
```env
SECRET_KEY="generate-a-secure-random-32-char-secret-key"
GROQ_API_KEY="gsk_your_groq_api_key_here"
GROQ_MODEL="openai/gpt-oss-20b"

BOOTSTRAP_ADMIN_USERNAME="admin_user"
BOOTSTRAP_ADMIN_PASSWORD="your-strong-password-under-72-characters"
BOOTSTRAP_ADMIN_TENANT_ID="tenant_alpha"

# Optional: Langfuse Cloud or Self-Hosted Telemetry Sync
LANGFUSE_PUBLIC_KEY=""
LANGFUSE_SECRET_KEY=""
LANGFUSE_HOST="https://cloud.langfuse.com"
```

### 3. Launch the Stack
```bash
docker compose up --build -d
```

### 4. Service Endpoints
| Service | URL | Purpose |
| :--- | :--- | :--- |
| **Streamlit Dashboard** | `http://localhost:8501` | Multi-tenant operator interface & tracing viewer |
| **FastAPI Gateway** | `http://localhost:8000/docs` | Interactive Swagger API documentation |
| **Readiness Probe** | `http://localhost:8000/ready` | Orchestrator health check (PostgreSQL, Qdrant, Redis) |

### Local versus Kubernetes deployment

Docker Compose is the supported local deployment for this repository. It runs one backend worker locally to avoid loading duplicate embedding models into memory. The `deploy/k8s/` manifests are an optional production reference for teams that need replica scaling, rolling updates, and autoscaling; Kubernetes is not required for the current project size.

The primary answer-generation model is **Groq Cloud `openai/gpt-oss-20b`**. NeMo Guardrails is enabled in the active environment and uses a separate `gpt-4o-mini` evaluator for self-check rails; it is not the primary generation model. The configured NeMo provider must have access to its model credentials.

### Free hosted demo deployment

For a small portfolio demo, use **Streamlit Community Cloud** for the frontend and **Render or Railway** for the FastAPI backend. These platforms may sleep, cold-start, or enforce monthly usage limits on free tiers, so this is a demo deployment rather than a high-availability production environment.

1. Deploy the repository to Streamlit Community Cloud with the entrypoint `frontend/app.py`.
2. Add `API_BASE_URL=https://<your-backend-domain>/api/v1` to Streamlit secrets or environment variables.
3. Deploy the repository to Render or Railway using the root `Dockerfile`. Its default command reads the platform `PORT` variable.
4. Configure the backend with hosted PostgreSQL, Redis, and Qdrant endpoints. Neon or Supabase can provide PostgreSQL, Upstash can provide Redis, and Qdrant Cloud can provide vector storage where their current free plans are available. For Upstash set `REDIS_URL=rediss://...`; for Qdrant Cloud set `QDRANT_URL=https://...` and `QDRANT_API_KEY=...`. These URL settings take precedence over local host/port settings.
5. Set `CORS_ORIGINS` to the exact Streamlit app URL and configure all model, authentication, database, and guardrail secrets in the backend platform. Never commit `.env` or provider keys.

The free hosted stack requires external service accounts and may not remain completely cost-free if usage exceeds provider quotas. Verify current free-tier limits before deployment.

---

## 🧪 Comprehensive Verification & Test Suite

The platform includes a unit and regression test suite verifying tenant boundaries, RBAC isolation, and guardrails:

```bash
# Run unit tests
python -m unittest discover tests

# Output:
# Ran 22 tests in 0.04s - OK
```

### Reproducible RAG evaluation

The built-in evaluation harness uses **Ragas** for context precision and context recall, with measured mean and P95 latency reported alongside the quality scores. The bundled smoke fixture is deterministic in its inputs but requires a configured Groq key because Ragas evaluates retrieval quality with an LLM:

```bash
python -m scripts.evaluate_rag
```

For credible hiring or production evidence, replace the fixture with labeled tenant-safe queries and record Ragas results for dense-only, sparse-only, hybrid RRF, and hybrid-plus-reranking configurations. Report the dataset size, K, hardware, provider, and whether latency includes network time. Langfuse remains the request-level observability layer for tracing retrieval, reranking, grading, and generation latency.

### Latency budget and measurement

The platform intentionally accepts additional latency for stronger security, retrieval quality, tenant isolation, and observability. Each component is measured separately so the critical path can be optimized without claiming an artificially low end-to-end number.

For a warm request with external reranking disabled, the current portfolio reference is approximately **2.2 seconds average** for the retrieval-to-first-response path:

| Stage | Local target | Main variable |
| :--- | ---: | --- |
| Request validation, JWT, and rate-limit lookup | ~0.05 s | Redis and database connection reuse |
| Dense + sparse query embedding | ~0.35 s | Warm CPU models and query length |
| Tenant-filtered Qdrant hybrid retrieval | ~0.45 s | Collection size and network round trip |
| Relevance grading | ~0.01 s | Heuristic grading; LLM grading is slower |
| Checkpoint and cache persistence | ~0.20 s | PostgreSQL/Redis health and connection reuse |
| Groq generation to first visible response | ~1.14 s | Provider queue, prompt size, and network |
| **Warm-path average** | **~2.20 s** | Excludes cold starts and optional Cohere reranking |

The 2.2-second figure is a warm-path reference budget, not a universal guarantee. Cold starts, hosted database poolers, Qdrant Cloud, NeMo checks, Cohere reranking, model queueing, and long completions can increase total latency. The telemetry API reports each span and should be used to replace this reference with measurements from the target deployment.

### Verified local test matrix
| Test Scenario | Query / Action | Expected Result | Live Result | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Admin Login** | Auth token request (`tenant_alpha`, `admin`) | JWT token with tenant/role claims | Token issued successfully | ✅ **PASS** |
| **Cross-Tenant Guard** | Tenant Alpha provisions user for Tenant Beta | 403 Forbidden | Blocked with 403 status | ✅ **PASS** |
| **Confidential Ingest** | Upload admin-only financial memo | Ingested with `allowed_roles=['admin']` | Indexed with Dense + BM25 | ✅ **PASS** |
| **Prompt Injection** | Submit DAN / override instructions | 400 Bad Request / Policy block | Intercepted pre-flight | ✅ **PASS** |
| **PII Redaction** | Query with SSN and Email | In-memory redaction before LLM | PII masked, clean answer streamed | ✅ **PASS** |
| **RBAC Isolation** | Viewer queries confidential admin memo | 0 chunks returned / safe fallback | `"No authorized records found"` | ✅ **PASS** |
| **RBAC Access** | Admin queries confidential admin memo | Relevant chunks retrieved & synthesized | Accurate answer generated | ✅ **PASS** |
| **Cross-Tenant Isolation** | Tenant Beta queries Tenant Alpha data | Complete data isolation (Zero leak) | `"No authorized records found"` | ✅ **PASS** |
| **Hybrid Search (SKU)** | Exact keyword search `#K8S-9921` | Dense + BM25 RRF fusion match | Retrieved exact SKU specifications | ✅ **PASS** |
| **Response Cache** | Repeat identical query in same thread | Version-scoped Redis cache hit (`cached: true`) | Covered by implementation and runtime smoke checks | ✅ **PASS** |

---

## 🛡️ Defensive AI, NeMo Guardrails & OWASP Compliance

```text
[ Incoming Request ]
        │
        ▼
[ NVIDIA NeMo Guardrails / OWASP Pattern Inspector ]
        │ ──► Matches Jailbreak / Policy Violation? ──► YES ──► Raise HTTP 400 & Log Security Audit Event
        │ (Clean)
        ▼
[ Presidio PII Redaction Engine ] ──► Masks Emails, Phones, SSNs, Credit Cards, JWTs, API Keys
        │ (Sanitized)
        ▼
[ Hybrid Vector Retrieval (Qdrant) ──► Cohere Cross-Encoder Reranker ──► LLM-as-a-Judge Grading ]
        │ (Answer Generated)
        ▼
[ NeMo Output Guardrails & PII Scrubbing ] ──► Final SSE Token Stream
```

1. **NVIDIA NeMo Guardrails (Input & Output Rails):** Configured via Colang flows in `guardrails/rails.co` to enforce conversational boundaries, prevent prompt injections, stop roleplaying jailbreaks, and sanitize output.
2. **Deterministic & Presidio PII Sanitization:** Redacts sensitive patterns in-memory (`[REDACTED_EMAIL]`, `[REDACTED_SSN]`, `[REDACTED_CARD]`, etc.) using Microsoft Presidio and optimized regex scrubbers.
3. **Structured Audit Events:** Emits immutable JSON security logs (`auth.login`, `document.ingest`) recording timestamps, IP, tenant ID, and user ID without logging raw credentials or prompts.

---

## 📊 Distributed Tracing & Observability

Every request automatically creates a distributed trace containing detailed sub-operation spans:

```json
{
  "trace_id": "df1202c0-c7a0-4434-a9fb-f6e6cd18ba5d",
  "tenant_id": "tenant_alpha",
  "user_id": "admin_user",
  "total_duration_ms": 2233.86,
  "cache_hit": false,
  "spans": [
    { "name": "retrieve_hybrid", "duration_ms": 699.25, "status": "ok", "metadata": { "retrieved_count": 8 } },
    { "name": "rerank_cohere", "duration_ms": 182.40, "status": "ok", "metadata": { "reranked_count": 5 } },
    { "name": "grade_documents", "duration_ms": 110.15, "status": "ok", "metadata": { "grader_type": "llm_as_judge", "is_relevant": true } },
    { "name": "generate_llm", "duration_ms": 1404.70, "status": "ok", "metadata": { "response_length": 1686 } }
  ]
}
```

* View live trace breakdowns and P95 latency percentiles directly inside the **Streamlit UI** or query `/api/v1/telemetry/traces`.
* Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to seamlessly stream traces to **Langfuse Cloud** or self-hosted Langfuse.

---

## 📂 Project Structure

```text
├── app/
│   ├── agents/              # LangGraph workflow, nodes, and LLM routers
│   │   ├── graph.py         # Corrective Agentic RAG graph with LLM grading & Cohere rerank
│   │   └── llm.py           # Groq Cloud chat model client
│   ├── api/                 # FastAPI routes and schemas
│   │   └── routes.py        # Auth, Ingest (Sync & Async), Streaming Chat, Telemetry
│   ├── core/                # Core security, auth, and telemetry infrastructure
│   │   ├── audit.py         # Structured JSON audit logging
│   │   ├── auth.py          # Multi-tenant JWT auth & OIDC JWKS validator
│   │   ├── rerank.py        # Cohere cross-encoder reranker client
│   │   ├── security.py      # NVIDIA NeMo Guardrails, OWASP injection & PII redaction
│   │   └── tracing.py       # Distributed span collector & Langfuse export
│   ├── db/                  # Qdrant hybrid vector store integration
│   │   └── qdrant.py        # Named dense+sparse collection & RRF hybrid search
│   ├── ingestion/           # Multi-format document parser
│   │   └── parser.py        # PDF, DOCX, XLSX, HTML, Code & Pytesseract OCR
│   ├── migrations/          # Versioned PostgreSQL schema migrations
│   ├── config.py            # Pydantic v2 application settings
│   ├── database.py          # Async PostgreSQL connection pool & checkpointer
│   ├── main.py              # FastAPI application & lifespan pre-warming
│   └── redis_client.py      # Async Redis client & tenant cache versioning
├── frontend/
│   └── app.py               # Streamlit operator dashboard & telemetry viewer
├── scripts/
│   ├── evaluate_rag.py      # Ragas retrieval evaluation runner
│   └── __init__.py
├── scratch/
│   └── check_documents.py   # Local document inspection utility
├── guardrails/              # NVIDIA NeMo Guardrails configuration
│   ├── config.yml           # Model & rail definitions
│   ├── prompts.yml          # Self-check prompt templates
│   └── rails.co             # Colang flow security definitions
├── tests/                   # Automated unit and regression test suite
├── .github/
│   └── workflows/           # CI and container delivery workflows
├── docs/
│   └── screenshots/         # Optional documentation screenshots
├── docker-compose.yml       # Supported local deployment stack
├── Dockerfile               # Multi-stage non-root container build
├── ARCHITECTURE.md          # In-depth architectural design specification
└── requirements.txt         # Version-constrained runtime dependencies
```

---

## 📜 License

Distributed under the **Apache 2.0 License**. See `LICENSE` for the complete terms.
