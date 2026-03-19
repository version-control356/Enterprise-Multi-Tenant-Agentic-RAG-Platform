"""Enterprise Streamlit dashboard for the Enterprise Multi-Tenant Agentic RAG Platform."""

import json
import logging
import os
import re
import time
import httpx
import streamlit as st
from html import escape, unescape

logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Enterprise Multi-Tenant Agentic RAG Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

_raw_api_url = os.getenv("API_BASE_URL", "").strip().rstrip("/")
if not _raw_api_url:
    try:
        _raw_api_url = str(st.secrets.get("API_BASE_URL", "")).strip().rstrip("/")
    except Exception:
        logger.debug("Streamlit secrets are unavailable; using the local API default.")
if not _raw_api_url:
    _raw_api_url = "http://localhost:8000/api/v1"

# Ensure scheme is present for cloud deployments (e.g. Render / Custom domains)
if not _raw_api_url.startswith(("http://", "https://")):
    if any(local_host in _raw_api_url for local_host in ("localhost", "127.0.0.1", "backend")):
        _raw_api_url = f"http://{_raw_api_url}"
    else:
        _raw_api_url = f"https://{_raw_api_url}"

if not _raw_api_url.endswith("/api/v1"):
    if _raw_api_url.endswith("/api"):
        API_BASE_URL = f"{_raw_api_url}/v1"
    else:
        API_BASE_URL = f"{_raw_api_url}/api/v1"
else:
    API_BASE_URL = _raw_api_url


def api_endpoint(path: str) -> str:
    """Safely build absolute API URLs without double slashes or missing prefix."""
    clean_path = path.lstrip("/")
    return f"{API_BASE_URL}/{clean_path}"


REQUEST_TIMEOUT = httpx.Timeout(60.0, connect=5.0)


@st.cache_data(ttl=15, show_spinner=False)
def check_backend_connection() -> tuple[bool, str, float]:
    """Test backend reachability and report latency."""
    t0 = time.perf_counter()
    try:
        health_url = API_BASE_URL.replace("/api/v1", "/health")
        resp = httpx.get(health_url, timeout=httpx.Timeout(4.0, connect=2.0), follow_redirects=True)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        if resp.is_success:
            return True, "Online", latency
        return False, f"HTTP {resp.status_code}", latency
    except Exception:
        latency = round((time.perf_counter() - t0) * 1000, 1)
        return False, "Unreachable", latency

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 0.4rem;
    }
    .badge-admin { background-color: #ede9fe; color: #6d28d9; border: 1px solid #c4b5fd; }
    .badge-analyst { background-color: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .badge-viewer { background-color: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
    .badge-tenant { background-color: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }
    .badge-cached { background-color: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
    .badge-trace { background-color: #f3f4f6; color: #374151; border: 1px solid #e5e7eb; font-family: monospace; }
    
    .doc-item {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .response-meta {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = None
if "tenant_id" not in st.session_state:
    st.session_state.tenant_id = "tenant_alpha"
if "username" not in st.session_state:
    st.session_state.username = None
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "default_session"
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []


def parse_jwt_claims(token: str) -> dict:
    """Extract claims from JWT payload for UI display."""
    try:
        import base64
        parts = token.split(".")
        if len(parts) >= 2:
            padded = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
            payload_bytes = base64.urlsafe_b64decode(padded)
            return json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        logger.debug("Unable to parse JWT claims.", exc_info=True)
    return {}


def clean_assistant_markdown(content: str) -> str:
    """Normalize model-generated HTML formatting for Streamlit Markdown."""
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
    return unescape(cleaned).strip()


def fetch_tenant_documents():
    """Load the list of indexed documents for the authenticated tenant."""
    if not st.session_state.jwt_token:
        st.session_state.uploaded_docs = []
        return
    try:
        resp = httpx.get(
            api_endpoint("documents"),
            headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
        )
        if resp.is_success:
            st.session_state.uploaded_docs = resp.json()
    except Exception:
        st.session_state.uploaded_docs = []


with st.sidebar:
    st.markdown("### 🛡️ Identity & Tenancy")
    
    is_online, conn_status, ping_ms = check_backend_connection()
    if is_online:
        st.caption(f"🟢 **Backend**: `{API_BASE_URL}` ({ping_ms} ms)")
    else:
        st.caption(f"🔴 **Backend ({conn_status})**: `{API_BASE_URL}`")
        st.info("💡 **Render Free Tier Notice**: Backend may be spinning up from sleep (~30s). Please wait a moment.")

    if not st.session_state.jwt_token:
        with st.form("login_form"):
            login_username = st.text_input("Username", value="admin_user")
            login_password = st.text_input("Password", type="password", value="your-strong-password-under-72-characters")
            login_tenant_id = st.text_input("Tenant ID", value="tenant_alpha")
            login_submitted = st.form_submit_button("🔑 Sign In", use_container_width=True)

        if login_submitted:
            try:
                auth_resp = httpx.post(
                    api_endpoint("auth/token"),
                    data={
                        "username": login_username,
                        "password": login_password,
                        "tenant_id": login_tenant_id,
                    },
                    timeout=REQUEST_TIMEOUT,
                    follow_redirects=True,
                )
                if auth_resp.is_success:
                    token = auth_resp.json()["access_token"]
                    claims = parse_jwt_claims(token)
                    st.session_state.jwt_token = token
                    st.session_state.tenant_id = login_tenant_id
                    st.session_state.username = claims.get("sub", login_username)
                    st.session_state.user_role = claims.get("role", "admin")
                    st.session_state.messages = []
                    fetch_tenant_documents()
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials or tenant ID.")
            except httpx.HTTPError as error:
                st.error(f"⚠️ Backend unavailable: {error}. If deployed on Render, backend may be waking up.")
    else:
        role_class = f"badge-{st.session_state.user_role}" if st.session_state.user_role in ["admin", "analyst", "viewer"] else "badge-trace"
        safe_username = escape(str(st.session_state.username or ""))
        safe_tenant_id = escape(str(st.session_state.tenant_id or ""))
        safe_role = escape(str(st.session_state.user_role or ""))
        st.markdown(f"""
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 0.5rem; padding: 0.8rem; margin-bottom: 0.8rem;">
            <div style="font-size: 0.8rem; color: #64748b;">Logged in as</div>
            <div style="font-size: 1rem; font-weight: 700; color: #0f172a;">{safe_username}</div>
            <div style="margin-top: 0.4rem;">
                <span class="badge-pill badge-tenant">🏢 {safe_tenant_id}</span>
                <span class="badge-pill {role_class}">👑 {safe_role}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            if st.button("➕ New Chat", use_container_width=True):
                st.session_state.thread_id = f"session_{int(time.time())}"
                st.session_state.messages = []
                st.rerun()
        with col_t2:
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.jwt_token = None
                st.session_state.username = None
                st.session_state.user_role = None
                st.session_state.messages = []
                st.session_state.uploaded_docs = []
                st.rerun()

        st.caption(f"💬 Active Thread: `{st.session_state.thread_id}`")

    if st.session_state.jwt_token and st.session_state.user_role == "admin":
        with st.expander("👥 Provision Tenant User", expanded=False):
            with st.form("user_provision_form"):
                member_username = st.text_input("Username")
                member_password = st.text_input("Password", type="password")
                member_role = st.selectbox("Role Clearance", options=["analyst", "viewer", "admin"])
                provision_submitted = st.form_submit_button("Create User", use_container_width=True)

            if provision_submitted:
                if not member_username or not member_password:
                    st.error("Please fill in username and password.")
                else:
                    try:
                        response = httpx.post(
                            api_endpoint("admin/users"),
                            headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
                            json={
                                "username": member_username,
                                "password": member_password,
                                "tenant_id": st.session_state.tenant_id,
                                "role": member_role,
                            },
                            timeout=REQUEST_TIMEOUT,
                            follow_redirects=True,
                        )
                        if response.status_code == 201:
                            st.success(f"✅ Created user `{member_username}` ({member_role})")
                        else:
                            st.error(response.json().get("detail", "User creation failed."))
                    except httpx.HTTPError as error:
                        st.error(f"Backend unavailable: {error}")

    st.divider()
    if st.session_state.user_role == "admin":
        try:
            resp = httpx.get(
                api_endpoint("admin/users"),
                headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
                timeout=REQUEST_TIMEOUT,
                follow_redirects=True,
            )
            users = resp.json() if resp.is_success else []
        except Exception:
            users = []
        if users:
            st.subheader("👥 Tenant Users")
            for u in users:
                col_u1, col_u2 = st.columns([3, 1])
                with col_u1:
                    st.markdown(f"- `{u['username']}` ({u['role']})")
                with col_u2:
                    if st.button("🗑️", key=f"del_user_{u['username']}"):
                        try:
                            del_resp = httpx.delete(
                                api_endpoint(f"admin/users/{u['username']}"),
                                headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
                                timeout=REQUEST_TIMEOUT,
                                follow_redirects=True,
                            )
                            if del_resp.is_success:
                                st.success(f"Deleted user `{u['username']}`")
                                st.rerun()
                            else:
                                st.error(f"Delete failed: {del_resp.text}")
                        except httpx.HTTPError as err:
                            st.error(f"Backend unavailable: {err}")
        else:
            st.caption("No other users in this tenant.")
    st.markdown("### 📄 Knowledge Base Ingestion")
    if st.session_state.jwt_token and st.session_state.user_role != "admin":
        st.info("🔒 Document ingestion is restricted to tenant administrators.")
    else:
        uploaded_file = st.file_uploader(
            "Upload Document (PDF, DOCX, XLSX, Code)",
            type=["pdf", "docx", "xlsx", "csv", "html", "txt", "md", "py", "js", "java", "cpp"],
            help="Files will be parsed with page markers, PII redacted, embedded with BGE Dense + BM25 Sparse vectors, and indexed into Qdrant."
        )
        selected_roles = st.multiselect(
            "Allowed Roles (RBAC Clearance)",
            options=["admin", "analyst", "viewer"],
            default=["admin"],
            help="Select which roles are permitted to search and retrieve from this document."
        )
        
        if uploaded_file and st.button("⚡ Index Document into Qdrant", use_container_width=True):
            if not st.session_state.jwt_token:
                st.error("Please sign in before uploading documents.")
            elif not selected_roles:
                st.error("Select at least one allowed role.")
            else:
                allowed_roles_str = ",".join(selected_roles)
                with st.spinner("Extracting text, redacting PII, and generating Dense+Sparse embeddings..."):
                    try:
                        response = httpx.post(
                            api_endpoint("ingest"),
                            headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
                            files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                            data={"allowed_roles": allowed_roles_str},
                            timeout=REQUEST_TIMEOUT,
                            follow_redirects=True,
                        )
                        if response.is_success:
                            res_data = response.json()
                            st.success(f"✅ Ingested **{res_data['chunks_ingested']} chunks** from `{uploaded_file.name}`.")
                            fetch_tenant_documents()
                            st.toast("Knowledge base updated and cache invalidated!", icon="🚀")
                            st.rerun()
                        else:
                            st.error(f"Ingestion failed: {response.text}")
                    except httpx.HTTPError as error:
                        st.error(f"Backend unavailable: {error}")

    if st.session_state.jwt_token:
        st.divider()
        st.markdown("### 📚 Indexed Documents")
        if not st.session_state.uploaded_docs:
            fetch_tenant_documents()

        if not st.session_state.uploaded_docs:
            st.caption("No documents indexed in this tenant yet.")
        else:
            for doc in st.session_state.uploaded_docs:
                size_kb = round(doc["size_bytes"] / 1024, 1)
                doc_roles = [r.strip() for r in doc.get("allowed_roles", "admin").split(",") if r.strip()]
                roles_badges_sb = " ".join(
                    f'<span class="badge-pill badge-{escape(r)}" style="font-size: 0.65rem; padding: 0.15rem 0.4rem;">{escape(r)}</span>'
                    for r in doc_roles
                )
                safe_filename = escape(str(doc.get("filename", "")))
                col_sd1, col_sd2 = st.columns([3, 1])
                with col_sd1:
                    st.markdown(f"""
                    <div class="doc-item">
                        <div style="font-weight: 600; color: #1e293b;">📄 {safe_filename}</div>
                        <div style="margin-top: 0.2rem;">{roles_badges_sb}</div>
                        <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.2rem;">
                            {doc['chunks_count']} chunks • {size_kb} KB
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_sd2:
                    if st.session_state.user_role == "admin":
                        if st.button("🗑️", key=f"sb_del_{doc['id']}", help=f"Delete {doc['filename']}"):
                            try:
                                resp = httpx.delete(
                                    api_endpoint(f"documents/{doc['id']}"),
                                    headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
                                    timeout=REQUEST_TIMEOUT,
                                    follow_redirects=True,
                                )
                                if resp.is_success:
                                    fetch_tenant_documents()
                                    st.toast(f"Deleted {doc['filename']}", icon="🗑️")
                                    st.rerun()
                                else:
                                    st.error(f"Delete failed ({resp.status_code}): {resp.text}")
                            except Exception as err:
                                st.error(f"Delete failed: {err}")


col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🛡️ Enterprise Multi-Tenant Agentic RAG Platform")
    st.caption("Zero-Trust Multi-Tenancy • Dense BGE + Sparse BM25 Hybrid Retrieval • Corrective LangGraph RAG • Langfuse Telemetry")

with col_h2:
    if st.session_state.jwt_token:
        safe_header_tenant = escape(str(st.session_state.tenant_id or ""))
        safe_header_role = escape(str(st.session_state.user_role or ""))
        st.markdown(f"""
        <div style="text-align: right; margin-top: 1rem;">
            <span class="badge-pill badge-tenant">Tenant: {safe_header_tenant}</span>
            <span class="badge-pill {role_class}">{safe_header_role}</span>
        </div>
        """, unsafe_allow_html=True)

tab_chat, tab_telemetry, tab_docs = st.tabs(["💬 Corrective RAG Chat", "📊 Observability & Distributed Traces", "📚 Knowledge Base Files"])

with tab_chat:
    col_c1, col_c2, col_c3, col_c4 = st.columns([3, 1, 1, 1])
    with col_c1:
        if st.session_state.jwt_token and st.session_state.uploaded_docs:
            doc_names_str = ", ".join(f"`{d['filename']}`" for d in st.session_state.uploaded_docs)
            st.markdown(f"**Indexed Knowledge Base:** {doc_names_str}")
        else:
            st.markdown("##### Enterprise Assistant (LangGraph State Machine)")
    with col_c2:
        if st.button("➕ New Thread", use_container_width=True):
            st.session_state.thread_id = f"session_{int(time.time())}"
            st.session_state.messages = []
            st.rerun()
    with col_c3:
        if st.button("🗑️ Clear Thread", use_container_width=True):
            if st.session_state.jwt_token:
                try:
                    httpx.delete(
                        api_endpoint(f"chat/thread/{st.session_state.thread_id}"),
                        headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
                        timeout=REQUEST_TIMEOUT,
                        follow_redirects=True,
                    )
                except httpx.HTTPError:
                    logger.debug("Thread history delete failed.", exc_info=True)
            st.session_state.messages = []
            st.toast("Active thread history cleared!", icon="🧹")
            st.rerun()
    with col_c4:
        if st.button("🧹 Purge All History", use_container_width=True):
            if st.session_state.jwt_token:
                try:
                    resp = httpx.delete(
                        api_endpoint("chat/history"),
                        headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
                        timeout=REQUEST_TIMEOUT,
                        follow_redirects=True,
                    )
                    if resp.is_success:
                        st.session_state.messages = []
                        st.session_state.thread_id = f"session_{int(time.time())}"
                        st.toast("All conversational checkpoints purged!", icon="🧹")
                        st.rerun()
                except Exception as err:
                    st.error(f"Failed to purge history: {err}")

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(
                    clean_assistant_markdown(message["content"])
                    if message["role"] == "assistant"
                    else message["content"]
                )
                if message.get("cached"):
                    st.markdown('<span class="badge-pill badge-cached">⚡ Instant Cache Hit (Redis)</span>', unsafe_allow_html=True)

    if prompt := st.chat_input("Ask questions against your uploaded documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            if not st.session_state.jwt_token:
                st.error("🔒 Please sign in from the sidebar before submitting a query.")
            else:
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    response_text = ""
                    is_cached = False
                    trace_id = None
                    t_start = time.perf_counter()

                    try:
                        with httpx.stream(
                            "POST",
                            api_endpoint("chat/stream"),
                            headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
                            params={
                                "query": prompt,
                                "thread_id": f"{st.session_state.tenant_id}_{st.session_state.thread_id}",
                            },
                            timeout=REQUEST_TIMEOUT,
                            follow_redirects=True,
                        ) as response:
                            if response.status_code == 400:
                                err_body = response.read().decode("utf-8", errors="ignore")
                                try:
                                    err_json = json.loads(err_body)
                                    err_detail = err_json.get("detail", "Security Policy Violation: Malicious prompt pattern detected.")
                                except Exception:
                                    err_detail = err_body
                                placeholder.warning(f"🚨 **Defensive AI Guardrail Interception:**\n\n{err_detail}")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"🚨 **Defensive AI Guardrail Interception:**\n\n{err_detail}"
                                })
                            elif not response.is_success:
                                err_body = response.read().decode("utf-8", errors="ignore")
                                placeholder.error(f"⚠️ Error ({response.status_code}): {err_body}")
                            else:
                                first_token_time = None
                                for line in response.iter_lines():
                                    if line.startswith("data: "):
                                        chunk_data = json.loads(line[6:])
                                        content_chunk = chunk_data.get("content", "")
                                        if content_chunk and first_token_time is None:
                                            first_token_time = time.perf_counter()
                                        response_text += content_chunk
                                        if chunk_data.get("cached"):
                                            is_cached = True
                                        if chunk_data.get("trace_id"):
                                            trace_id = chunk_data.get("trace_id")

                                        clean_display = clean_assistant_markdown(response_text)
                                        if clean_display:
                                            placeholder.markdown(f"{clean_display}▌")

                                dur_total = round((time.perf_counter() - t_start) * 1000, 1)
                                ttft_ms = round((first_token_time - t_start) * 1000, 1) if first_token_time else dur_total
                                final_clean = clean_assistant_markdown(response_text)
                                placeholder.markdown(final_clean)
                                
                                meta_badges = '<div class="response-meta">'
                                if is_cached:
                                    meta_badges += '<span class="badge-pill badge-cached">⚡ Redis Cache Hit</span>'
                                else:
                                    meta_badges += f'<span class="badge-pill badge-trace">⚡ TTFT: {ttft_ms} ms</span>'
                                    meta_badges += f'<span class="badge-pill badge-trace">⏱️ Stream: {dur_total} ms</span>'
                                if trace_id:
                                    meta_badges += f'<span class="badge-pill badge-trace">🔍 Trace: {trace_id[:8]}</span>'
                                meta_badges += '</div>'
                                st.markdown(meta_badges, unsafe_allow_html=True)

                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": final_clean,
                                    "cached": is_cached,
                                    "trace_id": trace_id,
                                })
                    except (httpx.HTTPError, json.JSONDecodeError) as error:
                        placeholder.error(f"Streaming error: {error}")


with tab_telemetry:
    col_t_head, col_t_btn1, col_t_btn2 = st.columns([2, 1, 1])
    with col_t_head:
        st.subheader("Tenant Performance & Distributed Traces")
        st.caption("Live latency tracking, cache efficiency, and discrete LangGraph span execution.")
    with col_t_btn1:
        refresh_telemetry = st.button("🔄 Refresh", use_container_width=True)
    with col_t_btn2:
        if st.button("🗑️ Clear Telemetry", use_container_width=True):
            if st.session_state.jwt_token:
                try:
                    resp = httpx.delete(
                        api_endpoint("telemetry/traces"),
                        headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
                        timeout=REQUEST_TIMEOUT,
                        follow_redirects=True,
                    )
                    if resp.is_success:
                        st.toast("Telemetry history cleared!", icon="🧹")
                        st.rerun()
                except Exception as err:
                    st.error(f"Failed to clear telemetry: {err}")

    if not st.session_state.jwt_token:
        st.info("🔑 Please sign in to view real-time observability and traces for your tenant.")
    else:
        try:
            metrics_resp = httpx.get(
                api_endpoint("telemetry/metrics"),
                headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
                timeout=REQUEST_TIMEOUT,
                follow_redirects=True,
            )
            if metrics_resp.is_success:
                metrics = metrics_resp.json()
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Requests", metrics.get("total_requests", 0))
                m2.metric("Cache Hit Rate", f"{metrics.get('cache_hit_rate', 0)}%")
                m3.metric("Avg Latency", f"{metrics.get('avg_latency_ms', 0)} ms")
                m4.metric("P95 Latency", f"{metrics.get('p95_latency_ms', 0)} ms")
                
                langfuse_status = "🟢 Active" if metrics.get("langfuse_enabled") else "⚪ Disabled"
                m5.metric("Langfuse Cloud Sync", langfuse_status)

                if metrics.get("langfuse_enabled"):
                    st.markdown("""
                    <div style="background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 0.5rem; padding: 0.6rem 1rem; margin-top: 0.5rem; font-size: 0.85rem; color: #1e40af; display: flex; align-items: center; justify-content: space-between;">
                        <span>🚀 Traces are syncing live to Langfuse Cloud with tenant and role tagging.</span>
                        <a href="https://cloud.langfuse.com" target="_blank" style="background-color: #2563eb; color: white; padding: 0.3rem 0.8rem; border-radius: 0.35rem; text-decoration: none; font-weight: 600; font-size: 0.8rem;">Open Langfuse Dashboard ↗</a>
                    </div>
                    """, unsafe_allow_html=True)

            st.divider()
            
            traces_resp = httpx.get(
                api_endpoint("telemetry/traces?limit=25"),
                headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
                timeout=REQUEST_TIMEOUT,
                follow_redirects=True,
            )
            if traces_resp.is_success:
                traces = traces_resp.json()
                if not traces:
                    st.info("No traces recorded yet for this tenant. Run a query in the Chat tab to generate traces.")
                else:
                    st.markdown(f"##### Recent Traces ({len(traces)})")
                    for t in traces:
                        status_icon = "🟢" if t["status"] == "ok" else "🔴"
                        cache_badge = "⚡ Cached" if t["cache_hit"] else f"⏱️ {t['total_duration_ms']} ms"
                        trace_title = f"{status_icon} Trace `{t['trace_id'][:8]}` — \"{t['query_preview']}\" ({cache_badge})"
                        
                        with st.expander(trace_title):
                            c1, c2, c3 = st.columns(3)
                            c1.markdown(f"**Tenant ID:** `{t['tenant_id']}`")
                            c2.markdown(f"**User & Role:** `{t['user_id']}` (`{t['role']}`)")
                            c3.markdown(f"**Rewrites:** `{t['rewrite_count']}` | **Cache Hit:** `{t['cache_hit']}`")
                            
                            if t.get("error_message"):
                                st.error(f"Trace Error: {t['error_message']}")

                            if t.get("spans"):
                                st.markdown("###### Execution Spans:")
                                for span in t["spans"]:
                                    span_status_icon = "✅" if span["status"] == "ok" else "❌"
                                    st.markdown(
                                        f"- {span_status_icon} **`{span['name']}`**: `{span['duration_ms']} ms` &nbsp;|&nbsp; "
                                        f"Metadata: `{json.dumps(span.get('metadata', {}))}`"
                                    )
        except httpx.HTTPError as error:
            st.error(f"Failed to fetch telemetry: {error}")


with tab_docs:
    col_d_head, col_d_action = st.columns([3, 1])
    with col_d_head:
        st.subheader("📚 Tenant Document Inventory")
        st.caption("All indexed files stored with hybrid embeddings in Qdrant and tracked in PostgreSQL.")
    with col_d_action:
        if st.session_state.jwt_token and st.session_state.user_role == "admin" and st.session_state.uploaded_docs:
            if st.button("🗑️ Clear All Documents", use_container_width=True):
                try:
                    resp = httpx.delete(
                        api_endpoint("documents"),
                        headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
                        timeout=REQUEST_TIMEOUT,
                        follow_redirects=True,
                    )
                    if resp.is_success:
                        fetch_tenant_documents()
                        st.toast("All indexed documents deleted from Qdrant & PostgreSQL!", icon="🧹")
                        st.rerun()
                except Exception as err:
                    st.error(f"Failed to clear documents: {err}")

    if not st.session_state.jwt_token:
        st.info("Please sign in to view your tenant's indexed documents.")
    else:
        fetch_tenant_documents()
        if not st.session_state.uploaded_docs:
            st.info("No documents have been indexed yet for this tenant. Upload documents using the sidebar form.")
        else:
            for doc in st.session_state.uploaded_docs:
                size_kb = round(doc["size_bytes"] / 1024, 1)
                doc_roles = [r.strip() for r in doc.get("allowed_roles", "admin").split(",") if r.strip()]
                roles_badges_html = " ".join(
                    f'<span class="badge-pill badge-{escape(r)}">{escape(r)}</span>' for r in doc_roles
                )
                safe_filename = escape(str(doc.get("filename", "")))
                safe_created_by = escape(str(doc.get("created_by", "")))
                safe_created_at = escape(str(doc.get("created_at", ""))[:19].replace("T", " "))
                col_card, col_btn = st.columns([5, 1])
                with col_card:
                    st.markdown(f"""
                    <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 0.5rem; padding: 1rem; margin-bottom: 0.8rem; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a;">📄 {safe_filename}</div>
                            <div>{roles_badges_html}</div>
                        </div>
                        <div style="display: flex; gap: 1.5rem; margin-top: 0.5rem; font-size: 0.85rem; color: #475569;">
                            <span>📊 <b>Chunks:</b> {doc['chunks_count']}</span>
                            <span>💾 <b>Size:</b> {size_kb} KB</span>
                            <span>👤 <b>Uploaded by:</b> {safe_created_by}</span>
                            <span>📅 <b>Uploaded at:</b> {safe_created_at}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_btn:
                    if st.session_state.user_role == "admin":
                        st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)
                        if st.button("🗑️ Delete", key=f"tab_del_{doc['id']}", use_container_width=True):
                            try:
                                resp = httpx.delete(
                                    api_endpoint(f"documents/{doc['id']}"),
                                    headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
                                    timeout=REQUEST_TIMEOUT,
                                    follow_redirects=True,
                                )
                                if resp.is_success:
                                    fetch_tenant_documents()
                                    st.toast(f"Deleted {doc['filename']}", icon="🗑️")
                                    st.rerun()
                                else:
                                    st.error(f"Delete failed ({resp.status_code}): {resp.text}")
                            except Exception as err:
                                st.error(f"Delete failed: {err}")
