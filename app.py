import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from config import ANTHROPIC_API_KEY, OPENAI_API_KEY, HF_TOKEN
from ingestion.loader import load_pdf
from indexing.indexers import OPENAI_EMBEDDING
from indexing.nonvectors import BM25_LANGUAGES
from query.handler import NO_STRATEGY
from generation.generator import LLM
import pipeline


# PAGE CONFIG ---------------------------------------------------------------
st.set_page_config(
    page_title="RAG Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# OPTIONS AND DEFAULTS ------------------------------------------------------
CHUNK_STRATEGIES = ["Fixed size", "Semantic", "Hierarchical"]
RETRIEVAL_MODES = ["Dense", "Sparse", "Hybrid"]
DENSE_MODELS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "BAAI/bge-large-en-v1.5", OPENAI_EMBEDDING]
OPENAI_EMBEDDING_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]
STOPWORD_MODES = ["None", "Language default", "Custom"]
QUERY_STRATEGIES = [NO_STRATEGY, "Rewrite", "HyDE", "Decompose"]
PROVIDERS = ["openai", "claude"]
MODELS = {
    "claude": ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
}

# Every setting lives in st.session_state under these keys; the widgets bind to them.
DEFAULTS = {
    "enable_generation": True,
    "chunk_strategy": CHUNK_STRATEGIES[0],
    "chunk_size": 500,
    "chunk_parent_size": 1000,
    "chunk_overlap": 30,
    "retrieval_mode": RETRIEVAL_MODES[0],
    "dense_model": DENSE_MODELS[0],
    "openai_embedding_model_name": OPENAI_EMBEDDING_MODELS[0],
    "bm25_language": BM25_LANGUAGES[0],
    "bm25_stemming": False,
    "bm25_stopwords_mode": STOPWORD_MODES[0],
    "bm25_stopwords_custom": "",
    "hybrid_dense_weight": 0.5,
    "retrieval_n_chunks": 5,
    "rerank": False,
    "rerank_candidates": 20,
    "query_strategy": QUERY_STRATEGIES[0],
    "provider": PROVIDERS[0],
    "model": MODELS[PROVIDERS[0]][0],
    "max_tokens": 4000,
    "anthropic_api_key": ANTHROPIC_API_KEY or "",
    "openai_api_key": OPENAI_API_KEY or "",
    "hf_token": HF_TOKEN or "",
}

# SESSION STATE -------------------------------------------------------------
session_id = get_script_run_ctx().session_id

# Streamlit drops the state of widgets that were not rendered in a run (e.g. the
# BM25 options while in Dense mode), so this also restores those to their default.
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

for key, val in {"index": None, "doc_name": None, "messages": [], "last_debug": None}.items():
    if key not in st.session_state:
        st.session_state[key] = val


def score_badge(chunk: dict) -> str:
    """Scores only get a traffic light when they are on a 0-1 relevance scale."""
    score, score_type = chunk["score"], chunk.get("score_type", "cosine")
    if score_type in ("cosine", "rerank"):
        icon = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
    else:
        icon = "⚪"
    return f"{icon} {score_type} {score:.2f}"


def retrieval_label(config: dict) -> str:
    """Short name of the retrieval setup, e.g. 'all-MiniLM-L6-v2' or 'BM25'."""
    parts = []
    if config["retrieval_mode"] in pipeline.DENSE_MODES:
        parts.append(config.get("openai_embedding_model_name") or config["dense_model"])
    if config["retrieval_mode"] in pipeline.SPARSE_MODES:
        parts.append("BM25")
    return " + ".join(parts)


def reset_model():
    st.session_state.model = MODELS[st.session_state.provider][0]


def index_document(uploaded, settings: dict):
    try:
        with st.status(f"Indexing {uploaded.name}...", expanded=True) as status:
            status.write("Reading PDF...")
            pages = load_pdf(uploaded.getvalue())
            index = pipeline.build_index(
                pages,
                pipeline.index_config(settings),
                session_id,
                openai_api_key=settings["openai_api_key"],
                hf_token=settings["hf_token"],
                log=status.write,
            )
    except Exception as e:
        st.error(f"Indexing failed: {e}")
        return

    st.session_state.index = index
    st.session_state.doc_name = uploaded.name
    st.session_state.messages = []
    st.session_state.last_debug = None
    st.rerun()  # redraw the sidebar with the new index


# SIDEBAR -------------------------------------------------------------------
with st.sidebar:
    st.title("🔍 RAG Explorer")

    api_key = st.session_state.anthropic_api_key if st.session_state.provider == "claude" else st.session_state.openai_api_key
    has_api_key = bool(api_key)
    current_config = pipeline.index_config({key: st.session_state[key] for key in DEFAULTS})
    index = st.session_state.index

    st.caption("Generation")
    st.toggle(
        "Enable LLM",
        key="enable_generation",
        disabled=not has_api_key,
        help="Off (or no API key): no LLM calls are made, you only see the retrieved chunks.",
    )
    llm_on = st.session_state.enable_generation and has_api_key

    st.caption("Document")
    with st.expander(st.session_state.doc_name or "Upload document", expanded=True):
        uploaded = st.file_uploader("Upload a PDF to start", type=["pdf"])

        missing_openai_key = current_config.get("dense_model") == OPENAI_EMBEDDING and not st.session_state.openai_api_key
        if missing_openai_key:
            st.warning("OpenAI embeddings need an OpenAI API key (see API keys below).")
        index_clicked = st.button(
            "Index document", type="primary", width="stretch",
            disabled=uploaded is None or missing_openai_key,
        )

        if index:
            st.success(f"✓ {st.session_state.doc_name}")
            st.caption(f"{index['chunk_count']} {index['config']['chunk_strategy']} chunks · {retrieval_label(index['config'])}")
            if current_config != index["config"]:
                st.warning("Chunking or retrieval settings changed since indexing. Index the document again to apply them.")

    st.caption("Chunking")
    chunk_side_label = f"{st.session_state.chunk_strategy} chunking (size={st.session_state.chunk_size}"
    if st.session_state.chunk_strategy == "Hierarchical":
        chunk_side_label += f", parent_size={st.session_state.chunk_parent_size}"
    chunk_side_label += f", overlap={st.session_state.chunk_overlap})"
    with st.expander(chunk_side_label, expanded=True):
        st.selectbox("Chunking strategy", CHUNK_STRATEGIES, key="chunk_strategy")
        st.number_input("Chunk size", min_value=1, key="chunk_size")
        st.number_input("Parent chunk size", min_value=2, key="chunk_parent_size", disabled=(st.session_state.chunk_strategy != "Hierarchical"))
        st.number_input("Chunk overlap", min_value=0, key="chunk_overlap")
        if st.session_state.chunk_overlap >= st.session_state.chunk_size:
            st.error("Chunk overlap must be smaller than chunk size.")
        if st.session_state.chunk_strategy == "Hierarchical" and st.session_state.chunk_parent_size <= st.session_state.chunk_size:
            st.error("Parent size must be larger than chunk size.")

    st.caption("Embedding and retrieval")
    with st.expander(f"{retrieval_label(current_config)} (n_chunks={st.session_state.retrieval_n_chunks})", expanded=True):
        st.selectbox("Retrieval mode", RETRIEVAL_MODES, key="retrieval_mode")

        if st.session_state.retrieval_mode in pipeline.DENSE_MODES:
            st.selectbox("Embedding model", DENSE_MODELS, key="dense_model")
            if st.session_state.dense_model == OPENAI_EMBEDDING:
                st.selectbox("OpenAI embedding model", OPENAI_EMBEDDING_MODELS, key="openai_embedding_model_name")

        if st.session_state.retrieval_mode in pipeline.SPARSE_MODES:
            st.selectbox("BM25 language", BM25_LANGUAGES, key="bm25_language")
            st.toggle("BM25 stemming", key="bm25_stemming")
            st.selectbox("BM25 stopwords", STOPWORD_MODES, key="bm25_stopwords_mode")
            if st.session_state.bm25_stopwords_mode == "Custom":
                st.text_input("Custom stopwords (comma separated)", key="bm25_stopwords_custom")

        if st.session_state.retrieval_mode == "Hybrid":
            st.slider(
                "Dense weight", min_value=0.0, max_value=1.0, step=0.05, key="hybrid_dense_weight",
                help="1.0 = only the embedding score counts, 0.0 = only BM25. Applies to the next question, no re-index needed.",
            )

        st.number_input("Chunks to retrieve", min_value=1, key="retrieval_n_chunks")

        st.toggle(
            "Rerank with cross-encoder", key="rerank",
            help="Retrieve more candidates, re-score each against the question with a cross-encoder "
                 "(ms-marco-MiniLM, ~90 MB download on first use) and keep the best. No re-index needed.",
        )
        if st.session_state.rerank:
            st.number_input("Candidates to rerank", min_value=1, key="rerank_candidates")

    st.caption("Query strategy")
    query_strategies_available = QUERY_STRATEGIES if llm_on else [NO_STRATEGY]
    if st.session_state.query_strategy not in query_strategies_available:
        st.session_state.query_strategy = NO_STRATEGY
    with st.expander(st.session_state.query_strategy, expanded=True):
        st.selectbox(
            "Query strategy", query_strategies_available, key="query_strategy",
            help=None if llm_on else "Rewrite, HyDE and Decompose need the LLM enabled.",
        )

    st.caption("Post retrieval")
    with st.expander("Upcoming feature!", expanded=False):
        st.checkbox("Corrective RAG", disabled=True)
        st.checkbox("Contextual compression", disabled=True)
        st.checkbox("Chunk positioning", disabled=True)

    st.caption("LLM")
    if st.session_state.model not in MODELS[st.session_state.provider]:
        reset_model()
    with st.expander(f"{st.session_state.model} (max_tokens={st.session_state.max_tokens})", expanded=True):
        st.selectbox("Provider", PROVIDERS, key="provider", on_change=reset_model)
        st.selectbox("Model", MODELS[st.session_state.provider], key="model")
        st.number_input("Max tokens", min_value=500, step=500, key="max_tokens")

    # Always rendered (never inside a condition) so Streamlit keeps the keys in session state
    st.caption("API keys")
    with st.expander("API keys", expanded=not has_api_key):
        st.text_input("Anthropic API key", type="password", key="anthropic_api_key")
        st.text_input("OpenAI API key", type="password", key="openai_api_key")
        st.text_input("Hugging Face token", type="password", key="hf_token", help="Only needed for gated or private embedding models.")

    st.caption("Evaluation")
    with st.expander("Evaluation", expanded=False):
        st.checkbox("Run RAGAS", disabled=True)

settings = {key: st.session_state[key] for key in DEFAULTS}

# Indexing runs after the whole sidebar is drawn: st.rerun() discards the state
# of every widget that was not rendered yet in the current run.
if index_clicked and uploaded:
    index_document(uploaded, settings)

# MAIN AREA --------------------------------------------------------------------
col_chat, col_debug = st.columns([3, 2])

# CHAT -------------------------------------------------------------------------
with col_chat:
    st.header("Chat")
    chat_container = st.container(height=600)

    if not index:
        question = st.chat_input("Upload and index a PDF first...", disabled=True)
    else:
        question = st.chat_input("Ask anything about your document...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Retrieving and generating..." if llm_on else "Retrieving..."):
                llm = LLM(settings["provider"], settings["model"], api_key) if llm_on else None
                result = pipeline.answer_question(question, index, settings, llm)
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "error": f"{type(e).__name__}: {e}"})
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "chunks": result["chunks"],
            })
            st.session_state.last_debug = result["debug"]

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                elif msg.get("error"):
                    st.error(msg["error"])
                else:
                    for chunk in msg["chunks"]:
                        with st.expander(f"{score_badge(chunk)} | 📍 Page {chunk['page']}"):
                            st.text(chunk["text"])
                    if not msg["chunks"]:
                        st.warning("No chunks matched this question.")
                    if msg["content"]:
                        st.markdown(msg["content"])
                    else:
                        st.info("LLM disabled or no API key, showing retrieved chunks only.")

# DEBUG PANE ---------------------------------------------------------------
with col_debug:
    st.header("🔬 Debug Panel")

    if st.session_state.last_debug is None:
        st.info("Ask a question to see debug info here.")
    else:
        debug = st.session_state.last_debug
        display = debug["query_display"]

        # costs
        if debug["llm_usage"]:
            with st.expander("💵 LLM tokens used (this question)", expanded=True):
                total_in = sum(u["input_tokens"] for u in debug["llm_usage"])
                total_out = sum(u["output_tokens"] for u in debug["llm_usage"])
                st.markdown(f"total: :green[{total_in:,}] in / :green[{total_out:,}] out")
                for u in debug["llm_usage"]:
                    st.markdown(f"{u['label']}: :green[{u['input_tokens']:,}] in / :green[{u['output_tokens']:,}] out")

        embedding_usage = index["dense"]["indexer"].usage if index and index["dense"] else []
        if embedding_usage:
            with st.expander("💵 Embedding tokens used (this document)", expanded=False):
                st.markdown(f"total: :green[{sum(u['tokens'] for u in embedding_usage):,}]")
                for u in embedding_usage:
                    st.markdown(f"{u['label']}: :green[{u['tokens']:,}]")

        # query trace
        with st.expander("💬 Query trace", expanded=False):
            st.markdown(f"**Strategy:** `{debug['query_strategy']}`")
            st.markdown(f"**Original:** {debug['question']}")

            if display.get("rewritten"):
                st.markdown("**Rewritten to:**")
                st.info(display["rewritten"])
            if display.get("hypothetical_document"):
                st.markdown("**HyDE — fake answer used for search:**")
                st.info(display["hypothetical_document"])
            if display.get("sub_questions"):
                st.markdown("**Decomposed into:**")
                for i, q in enumerate(display["sub_questions"], 1):
                    st.markdown(f"{i}. {q}")

        with st.expander(f"🔎 Retrieved chunks ({len(debug['chunks'])})", expanded=False):
            for i, chunk in enumerate(debug["chunks"], 1):
                st.markdown(f"**Chunk {i}** — Page {chunk['page']} — {score_badge(chunk)}")
                if chunk.get("score_type") == "rerank":
                    st.caption(f"↳ before reranking: {chunk['retrieval_score_type']} {chunk['retrieval_score']:.2f}")
                if chunk.get("score_type") == "hybrid":
                    dense_score = "not in dense results" if chunk["dense_score"] is None else f"cosine {chunk['dense_score']:.2f}"
                    sparse_score = "not in BM25 results" if chunk["sparse_score"] is None else f"bm25 {chunk['sparse_score']:.2f}"
                    st.caption(f"↳ {dense_score} · {sparse_score}")

                if display.get("sub_questions"):
                    for q, ids in debug["query_chunk_map"].items():
                        if chunk["chunk_id"] in ids:
                            st.caption(f"↳ from: {q}")

                if chunk.get("parent_text"):
                    tab_child, tab_parent = st.tabs(["Matched (child)", "Sent to LLM (parent)"])
                    with tab_child.container(height=120):
                        st.text(chunk["text"])
                    with tab_parent.container(height=120):
                        st.text(chunk["parent_text"])
                else:
                    with st.container(height=120):
                        st.text(chunk["text"])

        # process log
        with st.expander("🛠️ Process log", expanded=False):
            for i, s in enumerate(debug["steps"], 1):
                detail = f" — {s['detail']}" if s["detail"] else ""
                st.markdown(f"{i}. {'✅' if s['ran'] else '⬜'} {s['name']}{detail}")
