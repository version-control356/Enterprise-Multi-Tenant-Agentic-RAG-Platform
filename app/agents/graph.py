import operator
import asyncio
import logging
from functools import lru_cache
from typing import Annotated, TypedDict, List, Optional
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from app.config import settings
from app.agents.llm import get_llm_client
from app.db.qdrant import search_tenant_knowledge
from app.core.security import SecurityGuardrails
from app.core.tracing import telemetry_tracker
from app.core.rerank import rerank_documents_with_cohere
from fastembed import TextEmbedding, SparseTextEmbedding

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State schema for the Corrective Agentic RAG workflow."""
    messages: Annotated[List[BaseMessage], operator.add]
    tenant_id: str
    user_role: str
    context: str
    query: Optional[str]
    rewrite_count: int
    is_relevant: bool
    trace_id: Optional[str]
    available_documents: Optional[List[str]]


# Lazily initialize and share FastEmbed models
@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    """Lazily load and share the dense query embedding model."""
    return TextEmbedding("BAAI/bge-small-en-v1.5")


@lru_cache(maxsize=1)
def get_sparse_embedding_model() -> SparseTextEmbedding:
    """Lazily load and share the sparse BM25 query embedding model."""
    return SparseTextEmbedding("Qdrant/bm25")


def _extract_query(state: AgentState) -> str:
    """Extract user query from state messages or override query."""
    query = state.get("query")
    if query:
        return query
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
        elif hasattr(msg, "type") and msg.type in ("human", "user"):
            return str(msg.content)
        elif hasattr(msg, "content"):
            return str(msg.content)
    return ""


async def retrieve_node(state: AgentState) -> dict:
    """Retrieve relevant context from Qdrant using hybrid dense and sparse search.

    Args:
        state: The current agent state containing the user query, tenant ID and role.

    Returns:
        A dictionary with keys:
            - ``context``: The redacted text of the retrieved documents.
            - ``query``: The original user query (may be rewritten later).
            - ``rewrite_count``: Count of query rewrites performed so far.
            - ``is_relevant``: ``True`` if retrieval succeeded.
            - ``available_documents``: List of document filenames the tenant can access.
    """
    trace_id = state.get("trace_id") or ""
    user_query = _extract_query(state)
    if not user_query.strip():
        return {"context": "", "is_relevant": False, "available_documents": []}

    # Conversational greeting & courtesy fast path: skip heavy embedding computation and vector search
    user_query_clean = user_query.strip().lower()
    greetings_set = {"hi", "hello", "hey", "help", "who are you", "good morning", "good evening", "how are you", "thanks", "thank you"}
    if user_query_clean in greetings_set or (len(user_query_clean.split()) <= 2 and user_query_clean.startswith(("hi", "hello", "hey"))):
        return {
            "context": "",
            "query": user_query,
            "rewrite_count": 0,
            "is_relevant": True,
            "available_documents": [],
        }

    from app.database import list_tenant_documents

    doc_task = asyncio.create_task(
        list_tenant_documents(state["tenant_id"], user_role=state.get("user_role"))
    )
    async with telemetry_tracker.record_span(trace_id, "retrieve_hybrid", {"query": user_query[:50]}) as span_meta:
        # Extract dense and sparse embeddings concurrently
        dense_embs, sparse_embs = await asyncio.gather(
            asyncio.to_thread(lambda: list(get_embedding_model().embed([user_query]))),
            asyncio.to_thread(lambda: list(get_sparse_embedding_model().embed([user_query]))),
        )
        doc_records = await doc_task
        available_docs = [d["filename"] for d in doc_records] if doc_records else []

        query_dense_vector: List[float] = [float(x) for x in dense_embs[0]]
        query_sparse = sparse_embs[0]

        search_limit = 8 if (settings.USE_COHERE_RERANK or bool(settings.COHERE_API_KEY.strip())) else 5
        scored_points = await search_tenant_knowledge(
            dense_vector=query_dense_vector,
            tenant_id=state["tenant_id"],
            user_role=state["user_role"],
            sparse_indices=query_sparse.indices.tolist(),
            sparse_values=query_sparse.values.tolist(),
            limit=search_limit,
        )

        context_chunks = []
        for p in scored_points:
            if p.payload and "text" in p.payload:
                fname = p.payload.get("filename", "")
                text = str(p.payload.get("text", ""))
                context_chunks.append(f"--- Document Source: {fname} ---\n{text}")

        span_meta["retrieved_count"] = len(context_chunks)

        # Apply Cohere Cross-Encoder Reranking if configured
        if settings.USE_COHERE_RERANK or bool(settings.COHERE_API_KEY.strip()):
            async with telemetry_tracker.record_span(trace_id, "rerank_cohere", {"candidate_count": len(context_chunks)}) as rerank_meta:
                context_chunks = await rerank_documents_with_cohere(
                    query=user_query,
                    documents=context_chunks,
                    top_n=settings.COHERE_TOP_N,
                )
                rerank_meta["reranked_count"] = len(context_chunks)

        retrieved_context = SecurityGuardrails.redact_pii("\n\n".join(context_chunks))

    return {
        "context": retrieved_context,
        "query": user_query,
        "rewrite_count": state.get("rewrite_count", 0),
        "available_documents": available_docs,
    }


async def grade_documents_node(state: AgentState) -> dict:
    """Evaluates whether retrieved documents contain relevant context for the query with sub-millisecond fast path."""
    trace_id = state.get("trace_id") or ""
    async with telemetry_tracker.record_span(trace_id, "grade_documents") as span_meta:
        context = state.get("context", "").strip()
        if not context or len(context) < 30:
            span_meta["is_relevant"] = False
            span_meta["grader_type"] = "length_heuristic"
            return {"is_relevant": False}

        if not settings.USE_LLM_GRADER:
            user_query = _extract_query(state).lower()
            query_tokens = [w for w in user_query.split() if len(w) > 3]
            # Fast hybrid relevance: check for keyword or token presence in retrieved context
            has_overlap = any(token in context.lower() for token in query_tokens) if query_tokens else True
            span_meta["is_relevant"] = has_overlap
            span_meta["grader_type"] = "fast_heuristic_scorer"
            return {"is_relevant": has_overlap}

        try:
            llm = get_llm_client(temperature=0.0, max_tokens=3)
            user_query = _extract_query(state)
            grading_prompt = (
                "You are an enterprise document relevance grader evaluating retrieved context for a user question.\n"
                f"User Question: {user_query}\n\n"
                f"Retrieved Document Context:\n{context[:2000]}\n\n"
                "Task: Is the retrieved document context relevant to answering the question?\n"
                "Reply with ONLY a single word: 'yes' if relevant, or 'no' if irrelevant."
            )
            grading_response = await asyncio.wait_for(
                llm.ainvoke([SystemMessage(content=grading_prompt)]),
                timeout=4.0,
            )
            verdict = str(grading_response.content).strip().lower()
            is_rel = "yes" in verdict
            span_meta["is_relevant"] = is_rel
            span_meta["grader_type"] = "llm_as_judge"
            span_meta["llm_verdict"] = verdict[:20]
            return {"is_relevant": is_rel}
        except Exception as exc:
            logger.warning("LLM grading fallback triggered: %s", exc)
            is_rel = bool(len(context) >= 30)
            span_meta["is_relevant"] = is_rel
            span_meta["grader_type"] = "fallback_heuristic"
            return {"is_relevant": is_rel}


async def rewrite_query_node(state: AgentState) -> dict:
    """Reformulates the search query when initial retrieval produces insufficient context."""
    trace_id = state.get("trace_id") or ""
    async with telemetry_tracker.record_span(trace_id, "rewrite_query") as span_meta:
        llm = get_llm_client()
        original_query = _extract_query(state)
        docs_hint = ", ".join(state.get("available_documents") or [])
        prompt = (
            "You are an AI search query reformulation specialist.\n"
            f"Available document topics in knowledge base: {docs_hint or 'general technical'}\n"
            f"Original Question: {original_query}\n\n"
            "Rephrase and expand this into an optimized technical search query to retrieve relevant documentation.\n"
            "Output ONLY the revised search keywords/query without quotes or explanation."
        )
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        revised_query = str(response.content).strip()
        span_meta["original_query"] = original_query[:50]
        span_meta["revised_query"] = revised_query[:50]
        return {
            "query": revised_query,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }


def decide_to_generate(state: AgentState) -> str:
    """Conditional routing decision: generate answer or rewrite query for re-retrieval."""
    if not settings.ENABLE_QUERY_REWRITE:
        return "generate"
    if state.get("is_relevant") or state.get("rewrite_count", 0) >= 1:
        return "generate"
    if not state.get("available_documents"):
        return "generate"

    query = (state.get("query") or "").strip().lower()
    greetings = {"hi", "hello", "hey", "help", "who are you", "good morning", "good evening", "thanks", "thank you"}
    if query in greetings or len(query.split()) <= 2:
        return "generate"

    return "rewrite_query"


async def generate_node(state: AgentState) -> dict:
    """Generates response using context and the active LLM router."""
    trace_id = state.get("trace_id") or ""
    async with telemetry_tracker.record_span(trace_id, "generate_llm") as span_meta:
        llm = get_llm_client(temperature=0.2, max_tokens=1000)
        docs_list = state.get("available_documents", [])
        docs_summary = ", ".join(f"`{d}`" for d in docs_list) if docs_list else "No documents uploaded yet in this tenant."

        # Bound context length to 3000 chars for rapid LLM prefill and inference
        raw_context = state.get("context", "No relevant context found.")
        bounded_context = raw_context[:3000] if len(raw_context) > 3000 else raw_context

        user_query_raw = _extract_query(state).strip().lower()
        greetings_set = {"hi", "hello", "hey", "help", "who are you", "good morning", "good evening", "how are you", "thanks", "thank you"}
        is_greeting = user_query_raw in greetings_set or (len(user_query_raw.split()) <= 2 and user_query_raw.startswith(("hi", "hello", "hey")))

        if is_greeting:
            system_prompt = (
                "You are an intelligent, friendly enterprise AI assistant for a secure multi-tenant RAG platform.\n"
                "Respond cordially and concisely in 1-2 brief sentences, offering to assist with authorized workspace documentation.\n"
                "Do NOT dump document filenames or robotic error disclaimers for greetings."
            )
        elif settings.STRICT_RAG_MODE:
            system_prompt = (
                "You are an enterprise AI assistant operating under strict zero-trust knowledge grounding.\n\n"
                "CRITICAL CONSTRAINTS & GROUNDING RULES:\n"
                "1. Answer user questions ONLY and EXCLUSIVELY using the provided Document Context below.\n"
                "2. If the user explicitly asks what documents are available or uploaded in their workspace, list ONLY these authorized files: " + docs_summary + ".\n"
                "3. If the Document Context is empty, marked as 'No relevant context found.', or does not contain the specific facts needed to answer the question, state politely: 'Your authorized workspace documents do not contain information regarding this topic.' Do NOT dump or disclose document filenames unless explicitly asked.\n"
                "4. STRICT PROHIBITION: NEVER use pre-trained world knowledge, outside assumptions, or hallucinated facts to answer questions that are not present in the Document Context.\n"
                "5. Do NOT output internal reasoning, chain-of-thought, or <think> tags.\n\n"
                f"Document Context:\n{bounded_context}"
            )
        else:
            system_prompt = (
                "You are an intelligent, friendly, and expert enterprise AI assistant.\n\n"
                "GUIDELINES:\n"
                "1. If the user asks what documents/PDFs are uploaded or available, list ONLY the authorized documents shown here: " + docs_summary + ".\n"
                "2. Provide clear, comprehensive, and well-structured answers using the provided Document Context.\n"
                "3. Do NOT hallucinate unsupported facts. If a specific question cannot be answered from the provided documents, politely mention that no matching records were found in authorized files.\n"
                "4. Do NOT output internal reasoning, chain-of-thought, or <think> tags.\n\n"
                f"Document Context:\n{bounded_context}"
            )

        # Keep last 4 messages to preserve immediate context while bounding token usage
        recent_messages = state["messages"][-4:] if len(state["messages"]) > 4 else state["messages"]
        messages = [SystemMessage(content=system_prompt)] + recent_messages
        response = await llm.ainvoke(messages)
        span_meta["response_length"] = len(str(response.content))
        return {"messages": [response]}


def build_agent_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    """Constructs and compiles the Corrective Agentic RAG state graph with conditional loops."""
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        }
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile(checkpointer=checkpointer)
