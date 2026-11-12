import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from config import ANTHROPIC_API_KEY, OPENAI_API_KEY, HF_TOKEN
from ingestion.loader import load_pdf
from indexing.indexers import OPENAI_EMBEDDING
from indexing.nonvectors import BM25_LANGUAGES
from query.handler import NO_STRATEGY
from generation.generator import LLM
from generation import models as model_catalog
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
STOPWORD_MODES = ["None", "Language default", "Custom"]
QUERY_STRATEGIES = [NO_STRATEGY, "Rewrite", "HyDE", "Decompose"]
SAME_MODEL = "Same as answer model"
PROVIDERS = ["openai", "claude"]

# Every setting lives in st.session_state under these keys; the widgets bind to them.
DEFAULTS = {
    "enable_generation": True,
    "chunk_strategy": CHUNK_STRATEGIES[0],
    "chunk_size": 500,
    "chunk_parent_size": 1000,
    "chunk_overlap": 30,
    "retrieval_mode": RETRIEVAL_MODES[0],
    "dense_model": DENSE_MODELS[0],
    "openai_embedding_model_name": model_catalog.FALLBACK_EMBEDDING_MODELS[0],
    "bm25_language": BM25_LANGUAGES[0],
    "bm25_stemming": False,
    "bm25_stopwords_mode": STOPWORD_MODES[0],
    "bm25_stopwords_custom": "",
    "question_index": False,
    "questions_per_chunk": 2,
    "hybrid_dense_weight": 0.5,
    "retrieval_n_chunks": 5,
    "rerank": False,
    "rerank_candidates": 20,
    "query_strategy": QUERY_STRATEGIES[0],
    "corrective_rag": False,
    "contextual_compression": False,
    "chunk_positioning": False,
    "post_retrieval_model": SAME_MODEL,
    "provider": PROVIDERS[0],
    "model": model_catalog.FALLBACK_CHAT_MODELS[PROVIDERS[0]][0],
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


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_chat_models(provider: str, api_key: str) -> list[str]:
    return model_catalog.fetch_chat_models(provider, api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_embedding_models(api_key: str) -> list[str]:
    return model_catalog.fetch_embedding_models(api_key)


def live_or_fallback(fetch, api_key: str, fallback: list[str]) -> tuple[list[str], str]:
    """(options, note about where they came from). Lists are fetched from the
    provider once an hour per key; failures are not cached, so a fixed key
    works on the next rerun."""
    if not api_key:
        return fallback, "Built-in list. Add an API key to load the current models."
    try:
        return fetch(api_key) or fallback, "Live list from the provider, refreshed hourly."
    except Exception as e:
        return fallback, f"Built-in list, could not load the live one ({type(e).__name__})."


def select_model(label: str, key: str, options: list[str], fallback: list[str], note: str):
    """Selectbox over a model list that can change between runs."""
    if st.session_state[key] not in options:
        st.session_state[key] = model_catalog.default_model(options, fallback)
    st.selectbox(label, options, key=key, help=note)


def usage_summary(usage: list[dict]) -> list[str]:
    """LLM token usage as markdown lines: a total, then one line per kind of call
    and model."""
    by_label = {}
    for u in usage:
        label = f"{u['label']} · `{u['model']}`"
        entry = by_label.setdefault(label, {"calls": 0, "in": 0, "out": 0})
        entry["calls"] += 1
        entry["in"] += u["input_tokens"]
        entry["out"] += u["output_tokens"]
    total_in, total_out = sum(e["in"] for e in by_label.values()), sum(e["out"] for e in by_label.values())
    lines = [f"total: :green[{total_in:,}] in / :green[{total_out:,}] out"]
    for label, e in by_label.items():
        calls = f" ({e['calls']} calls)" if e["calls"] > 1 else ""
        lines.append(f"{label}{calls}: :green[{e['in']:,}] in / :green[{e['out']:,}] out")
    return lines


def index_document(uploaded, settings: dict, llm):
    try:
        with st.status(f"Indexing {uploaded.name}...", expanded=True) as status:
            status.write("Reading PDF...")
            pages = load_pdf(uploaded.getvalue())
            config = pipeline.index_config(settings)
            progress_bar = st.progress(0.0) if config.get("question_index") else None
            index = pipeline.build_index(
                pages,
                config,
                session_id,
                openai_api_key=settings["openai_api_key"],
                hf_token=settings["hf_token"],
                llm=llm,
                log=status.write,
                progress=progress_bar.progress if progress_bar else None,
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
    if not (has_api_key and st.session_state.enable_generation):
        st.session_state.question_index = False  # needs the LLM, see the toggle below
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
            if index["config"].get("question_index"):
                st.caption(f"Indexed as {index['config']['questions_per_chunk']} hypothetical questions per chunk")
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
                embedding_models, note = live_or_fallback(
                    _fetch_embedding_models, st.session_state.openai_api_key, model_catalog.FALLBACK_EMBEDDING_MODELS,
                )
                select_model("OpenAI embedding model", "openai_embedding_model_name", embedding_models, model_catalog.FALLBACK_EMBEDDING_MODELS, note)

        if st.session_state.retrieval_mode in pipeline.SPARSE_MODES:
            st.selectbox("BM25 language", BM25_LANGUAGES, key="bm25_language")
            st.toggle("BM25 stemming", key="bm25_stemming")
            st.selectbox("BM25 stopwords", STOPWORD_MODES, key="bm25_stopwords_mode")
            if st.session_state.bm25_stopwords_mode == "Custom":
                st.text_input("Custom stopwords (comma separated)", key="bm25_stopwords_custom")

        st.toggle(
            "Index hypothetical questions", key="question_index", disabled=not llm_on,
            help="Instead of the chunk text, index questions an LLM writes for each chunk, so your question is "
                 "matched against questions. Costs one LLM call per chunk when indexing. HyDE is unavailable with it.",
        )
        if st.session_state.question_index:
            st.number_input("Questions per chunk", min_value=1, max_value=5, key="questions_per_chunk")

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
    if index and index["config"].get("question_index"):
        # the index holds questions, so turning the query into a fake answer would hurt
        query_strategies_available = [s for s in query_strategies_available if s != "HyDE"]
    if st.session_state.query_strategy not in query_strategies_available:
        st.session_state.query_strategy = NO_STRATEGY
    with st.expander(st.session_state.query_strategy, expanded=True):
        st.selectbox(
            "Query strategy", query_strategies_available, key="query_strategy",
            help=None if llm_on else "Rewrite, HyDE and Decompose need the LLM enabled.",
        )

    fallback_models = model_catalog.FALLBACK_CHAT_MODELS[st.session_state.provider]
    chat_models, models_note = live_or_fallback(
        lambda key: _fetch_chat_models(st.session_state.provider, key), api_key, fallback_models,
    )

    st.caption("Post retrieval")
    if st.session_state.post_retrieval_model not in chat_models:  # provider or list changed
        st.session_state.post_retrieval_model = SAME_MODEL
    if not llm_on:
        st.session_state.corrective_rag = False
        st.session_state.contextual_compression = False
    post_steps = [
        name for name, key in (("Corrective RAG", "corrective_rag"), ("Compression", "contextual_compression"), ("Positioning", "chunk_positioning"))
        if st.session_state[key]
    ]
    with st.expander(" + ".join(post_steps) or "None", expanded=False):
        st.checkbox(
            "Corrective RAG", key="corrective_rag", disabled=not llm_on,
            help="The LLM judges each retrieved chunk and drops the ones that do not help answer the question. One extra LLM call.",
        )
        st.checkbox(
            "Contextual compression", key="contextual_compression", disabled=not llm_on,
            help="The LLM cuts each chunk down to the sentences relevant to the question. One extra LLM call per chunk.",
        )
        st.checkbox(
            "Chunk positioning", key="chunk_positioning",
            help="Order the chunks so the best ones are first and last in the prompt and the weakest sit in the middle.",
        )
        st.selectbox(
            "Model for corrective RAG and compression", [SAME_MODEL] + chat_models, key="post_retrieval_model",
            disabled=not llm_on,
            help="These steps make many small judgement calls, so a cheaper model than the one writing the answer is often enough.",
        )

    st.caption("LLM")
    if st.session_state.model not in chat_models:  # provider or list changed
        st.session_state.model = model_catalog.default_model(chat_models, fallback_models)
    with st.expander(f"{st.session_state.model} (max_tokens={st.session_state.max_tokens})", expanded=True):
        st.selectbox("Provider", PROVIDERS, key="provider")
        select_model("Model", "model", chat_models, fallback_models, models_note)
        st.caption(models_note)
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
    index_document(uploaded, settings, LLM(settings["provider"], settings["model"], api_key) if settings["question_index"] else None)

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
                post_llm = None
                if llm_on and settings["post_retrieval_model"] not in (SAME_MODEL, settings["model"]):
                    post_llm = LLM(settings["provider"], settings["post_retrieval_model"], api_key)
                result = pipeline.answer_question(question, index, settings, llm, post_llm)
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
                for line in usage_summary(debug["llm_usage"]):
                    st.markdown(line)

        if index and index["llm_usage"]:
            with st.expander("💵 LLM tokens used (indexing this document)", expanded=False):
                for line in usage_summary(index["llm_usage"]):
                    st.markdown(line)

        embedding_usage = index["dense"]["indexer"].usage if index and index["dense"] else []
        if embedding_usage:
            with st.expander("💵 Embedding tokens used (this document)", expanded=False):
                embedding_model = index["config"].get("openai_embedding_model_name", index["config"]["dense_model"])
                st.markdown(f"model: `{embedding_model}`")
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
                if chunk.get("matched_question"):
                    st.caption(f"↳ matched question: {chunk['matched_question']}")
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

                # one tab per version of the chunk; the last one is what the LLM received
                versions = {"Matched": chunk["text"]}
                if chunk.get("parent_text"):
                    versions = {"Matched (child)": chunk["text"], "Parent": chunk["parent_text"]}
                if chunk.get("compressed_text"):
                    versions["Compressed"] = chunk["compressed_text"]
                if len(versions) == 1:
                    with st.container(height=120):
                        st.text(chunk["text"])
                else:
                    labels = list(versions)
                    labels[-1] += " → sent to LLM"
                    for tab, text in zip(st.tabs(labels), versions.values()):
                        with tab.container(height=120):
                            st.text(text)

        if debug["dropped_chunks"]:
            with st.expander(f"🗑️ Dropped by Corrective RAG ({len(debug['dropped_chunks'])})", expanded=False):
                for chunk in debug["dropped_chunks"]:
                    st.markdown(f"Page {chunk['page']} — {score_badge(chunk)}")
                    with st.container(height=120):
                        st.text(chunk["text"])

        # process log
        with st.expander("🛠️ Process log", expanded=False):
            for i, s in enumerate(debug["steps"], 1):
                detail = f" — {s['detail']}" if s["detail"] else ""
                st.markdown(f"{i}. {'✅' if s['ran'] else '⬜'} {s['name']}{detail}")
