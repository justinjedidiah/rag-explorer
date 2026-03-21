import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import tempfile
import os
from config import ANTHROPIC_API_KEY, OPENAI_API_KEY
from ingestion.loader import load_pdf
from ingestion.chunker import chunk_documents
from indexing.indexers import get_indexer
from indexing.vectorstore import load_index, cleanup_old_collections
from indexing.vectors import SentenceTransformerIndexerParams, OpenAIIndexerParams
from indexing.nonvectors import BM25IndexerParams, STOPWORDS_LANG_BM25
from retrieval.retriever import retrieve
from generation.generator import generate
from query.handler import handle_query
from generation.generator import quick_llm


# PAGE CONFIG ---------------------------------------------------------------
st.set_page_config(
    page_title="RAG Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SESSION SETUPS ------------------------------------------------------------

session_id = get_script_run_ctx().session_id
cleanup_old_collections(max_age_hours=24)
# debug state if needed
# st.write({k: v for k, v in dict(st.session_state).items() if k not in ('anthropic_api_key','openai_api_key','_tmp_anthropic_api_key','_tmp_openai_api_key')})

if "collection" not in st.session_state:
    st.session_state.collection = load_index(session_id)  # restore on refresh

if "indexer" not in st.session_state:
    st.session_state.indexer = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

if "last_debug" not in st.session_state:
    st.session_state.last_debug = None

chunk_strategy_options = ["Fixed size","Semantic","Hierarchical"]
retrieval_mode_options = ["Dense","Sparse"]
dense_retrieval_model_options = ["all-MiniLM-L6-v2","BAAI/bge-large-en-v1.5","OpenAI Embedding"]
openai_embedding_model_options = ["text-embedding-3-small","text-embedding-3-large"]
sparse_retrieval_model_options = ["BM25"]
query_strategy_options = ["None (raw question)", "Rewrite", "HyDE", "Decompose"]
provider_options = ["openai","claude"]
model_options_claude = ["claude-haiku-4-5-20251001","claude-sonnet-4-6","claude-opus-4-6"]
model_options_openai = ["gpt-4o-mini", "gpt-4o"]

DEFAULTS = {
    "enable_generation": True,
    "chunk_strategy": chunk_strategy_options[0],
    "chunk_size": 500,
    "chunk_parent_size": 1000,
    "chunk_overlap": 30,
    "chunk_count": None,
    "retrieval_mode": retrieval_mode_options[0],
    "retrieval_model": dense_retrieval_model_options[0], # dense because retrieval mode default is dense
    "openai_embedding_model_name": openai_embedding_model_options[0],
    "bm25_params": {"stemming": False, "language": None, "stopwords":None},
    "retrieval_n_chunks": 5,
    "query_strategy": query_strategy_options[0],
    "provider": provider_options[0],
    "anthropic_api_key": ANTHROPIC_API_KEY or None,
    "openai_api_key": OPENAI_API_KEY or None,
    "model": model_options_openai[0],
    "max_tokens": 4000,
}

# initialize states
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

if "prev_provider" not in st.session_state:
    st.session_state.prev_provider = st.session_state.provider

if st.session_state.provider != st.session_state.prev_provider:
    if st.session_state.provider == "claude":
        st.session_state.model = model_options_claude[0]
    elif st.session_state.provider == "openai":
        st.session_state.model = model_options_openai[0]
    st.session_state.prev_provider = st.session_state.provider

with st.sidebar:
    st.title("🔍 RAG Explorer")

    st.caption("Generation")
    api_key = st.session_state.anthropic_api_key if st.session_state.provider == "claude" else st.session_state.openai_api_key
    has_api_key = bool(api_key)
    st.toggle("Enable LLM Generation", 
        key="enable_generation",
        disabled=not has_api_key
    )

    st.caption("Document")
    doc_side_label = f"{st.session_state.doc_name or "Upload document"}"
    with st.expander(doc_side_label, expanded=True):
        uploaded = st.file_uploader("Upload a PDF to start", type=["pdf"])

        if uploaded:
            if st.button("Index Document", type="primary", use_container_width=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                with st.spinner("Loading PDF..."):
                    pages = load_pdf(tmp_path)
                with st.spinner(f"Chunking ({len(pages)} pages)..."):
                    if st.session_state.chunk_strategy == "Hierarchical" and st.session_state.chunk_parent_size <= st.session_state.chunk_size:
                        st.error("Parent size must be larger than chunk size.")
                        st.stop()
                    chunks = chunk_documents(pages, st.session_state.chunk_strategy, size=st.session_state.chunk_size, overlap=st.session_state.chunk_overlap, parent_size=st.session_state.chunk_parent_size)
                with st.spinner(f"Embedding ({len(chunks)} chunks)..."):
                    if st.session_state.retrieval_mode == "Dense":
                        if st.session_state.retrieval_model == "OpenAI Embedding":
                            if not st.session_state.openai_api_key:
                                st.warning("To use openai embedding models, you need to input an api key in the generation settings")
                                st.text_input(
                                    "OpenAI API key",
                                    type="password",
                                    key="openai_api_key",
                                    on_change=st.rerun
                                )
                                st.stop()
                            retrieval_params: OpenAIIndexerParams = {
                                "model_name": st.session_state.openai_embedding_model_name,
                                "api_key": st.session_state.openai_api_key,
                            }
                        else:
                            retrieval_params: SentenceTransformerIndexerParams = {
                                "model_name": st.session_state.retrieval_model,
                                "hf_token": None, # TODO add hf token input
                            }
                    elif st.session_state.retrieval_mode == "Sparse":
                        if st.session_state.retrieval_model == "BM25":
                            retrieval_params : BM25IndexerParams = st.session_state.bm25_params
                    indexer = get_indexer(
                        retrieval_mode=st.session_state.retrieval_mode,
                        model_key=st.session_state.retrieval_model,
                        params=retrieval_params
                    )
                    processed_chunks = indexer.process_chunks(chunks)
                with st.spinner("Building index..."):
                    if st.session_state.retrieval_mode == "Dense":
                        build_index_params = {
                            "chunks": chunks,
                            "vectors": processed_chunks,
                            "session_id": session_id
                        }
                    elif st.session_state.retrieval_mode == "Sparse":
                        if st.session_state.retrieval_model == "BM25":
                            build_index_params = {
                                "chunks": chunks,
                                "corpus_tokens": processed_chunks
                            }
                    collection = indexer.build_index(**build_index_params)

                os.unlink(tmp_path)
                st.session_state.collection = collection
                st.session_state.indexer = indexer
                st.session_state.doc_name = uploaded.name
                st.session_state.chunk_count = len(chunks)
                st.session_state.messages = []
                st.session_state.last_debug = None

                # rescue states before rerun, some states will be missing after rerun because it only keeps states used in expander for some reason
                for key, val in DEFAULTS.items():
                    if key != "enable_generation":  # widget manages this itself
                        st.session_state[key] = st.session_state.get(key, val)

                st.rerun()

        if st.session_state.collection:
            st.success(f"✓ {st.session_state.doc_name or 'Document indexed'}")
            st.caption(f"{st.session_state.chunk_count} {st.session_state.chunk_strategy} chunks")

    st.caption("Chunking")
    chunk_side_label = f"{st.session_state.chunk_strategy} chunking (size={st.session_state.chunk_size}{f", parent_size={st.session_state.chunk_parent_size}" if st.session_state.chunk_strategy == "Hierarchical" else ""}, overlap={st.session_state.chunk_overlap})"
    with st.expander(chunk_side_label, expanded=True):
        st.selectbox("Chunking Strategy", chunk_strategy_options, key="chunk_strategy")
        st.number_input("Chunk Size", min_value=1, key='chunk_size')
        st.number_input("Parent Chunk Size", min_value=2, key="chunk_parent_size", disabled=(st.session_state.chunk_strategy != "Hierarchical"))
        st.number_input("Chunk Overlap", min_value=0, key='chunk_overlap')

    st.caption("Embedding and Retrieval")
    retrieval_side_name = f"{st.session_state.retrieval_model} (n_chunks={st.session_state.retrieval_n_chunks})"
    with st.expander(retrieval_side_name, expanded=True):
        st.selectbox("Retrieval Mode", retrieval_mode_options, key="retrieval_mode")

        if st.session_state.retrieval_mode == "Dense":
            st.session_state.bm25_params = DEFAULTS["bm25_params"]
            retrieval_model_options = dense_retrieval_model_options
        elif st.session_state.retrieval_mode == "Sparse":
            retrieval_model_options = sparse_retrieval_model_options
        st.selectbox("Model", retrieval_model_options, key="retrieval_model")

        if st.session_state.retrieval_model == "OpenAI Embedding":
            st.selectbox("OpenAI Embedding Models", openai_embedding_model_options, key="openai_embedding_model_name")

        elif st.session_state.retrieval_model == "BM25":
            st.session_state.bm25_params.update({"language": st.session_state.get("_tmp_bm25_language", STOPWORDS_LANG_BM25[0])})
            st.selectbox("BM25 Language", STOPWORDS_LANG_BM25, key="_tmp_bm25_language",
                on_change=lambda: st.session_state.bm25_params.update({"language": st.session_state._tmp_bm25_language})
            )
            st.session_state.bm25_params.update({"stemming": st.session_state.get("_tmp_bm25_stemming", False)})
            st.toggle("BM25 Stemming", key="_tmp_bm25_stemming",
                on_change=lambda: st.session_state.bm25_params.update({"stemming": st.session_state._tmp_bm25_stemming})
            )
            def clear_bm25_stopwords():
                if st.session_state._tmp_bm25_stopwords_toggle:
                    st.session_state.bm25_params.update({"stopwords": st.session_state.get("_tmp_bm25_stopwords_defined", STOPWORDS_LANG_BM25[0])})
                else:
                    st.session_state.bm25_params.update({"stopwords": None})
            st.toggle("BM25 Stopwords", key="_tmp_bm25_stopwords_toggle", on_change=clear_bm25_stopwords)
            if st.session_state._tmp_bm25_stopwords_toggle:
                st.toggle("Custom?", key="_tmp_bm25_stopwords_toggle_iscustom",on_change=clear_bm25_stopwords)
                if not st.session_state._tmp_bm25_stopwords_toggle_iscustom:
                    st.session_state.bm25_params.update({"stopwords": st.session_state.get("_tmp_bm25_stopwords_defined", STOPWORDS_LANG_BM25[0])})
                    st.selectbox("BM25 Stopwords", STOPWORDS_LANG_BM25, key="_tmp_bm25_stopwords_defined",
                        on_change=lambda: st.session_state.bm25_params.update({"stopwords": st.session_state._tmp_bm25_stopwords_defined})
                    )
                else:
                    def split_stopwords(text, separator):
                        splitted = text.split(separator)
                        return [t.strip() for t in splitted]
                    st.session_state.bm25_params.update({
                        "stopwords": split_stopwords(
                            st.session_state.get("_tmp_bm25_stopwords_custom", ""),
                            st.session_state.get("_tmp_bm25_stopwords_custom_separator", ",")
                        )
                    })
                    st.text_input("Separator", value=",", key="_tmp_bm25_stopwords_custom_separator")
                    st.text_input("BM25 Stopwords", key="_tmp_bm25_stopwords_custom",
                        on_change=lambda: st.session_state.bm25_params.update({
                            "stopwords":(
                                split_stopwords(
                                    st.session_state._tmp_bm25_stopwords_custom,
                                    st.session_state._tmp_bm25_stopwords_custom_separator
                                )
                            )
                        })
                    )

        st.number_input("Chunks to Retrieve", min_value=1, key="retrieval_n_chunks")

    st.caption("Query Strategy")
    query_strategy_options_available = ["None (raw question)"]
    if st.session_state.enable_generation and has_api_key:
        query_strategy_options_available = query_strategy_options  # all options
    query_strategy_side_name = f"{st.session_state.query_strategy}"
    with st.expander(query_strategy_side_name, expanded=True):
        st.selectbox("Query Strategy", query_strategy_options_available, key="query_strategy")

    st.caption("Post retrieval")
    post_retrieval_side_name = "Upcoming feature!"
    with st.expander(post_retrieval_side_name, expanded=False):
        st.checkbox("Corrective RAG", disabled=True)
        st.checkbox("Contextual compression", disabled=True)
        st.checkbox("Chunk positioning", disabled=True)

    st.caption("LLM")
    model_side_name = f"{st.session_state.model} (max_tokens={st.session_state.max_tokens})"
    with st.expander(model_side_name, expanded=True):
        st.selectbox("Provider", ["claude", "openai"], key="provider")

        if st.session_state.provider == "claude":
            st.selectbox("Model", model_options_claude, key="model")
            st.number_input("Max Tokens", min_value=500, step=500, key="max_tokens")
            st.text_input(
                "Anthropic API key",
                type="password",
                key="_tmp_anthropic_api_key",
                value=st.session_state.anthropic_api_key,
                on_change=lambda: setattr(
                    st.session_state, 
                    "anthropic_api_key", 
                    st.session_state._tmp_anthropic_api_key
                )
            )
        elif st.session_state.provider == "openai":
            st.selectbox("Model", model_options_openai, key="model")
            st.number_input("Max Tokens", min_value=500, step=500, key="max_tokens")
            st.text_input(
                "OpenAI API key",
                type="password",
                key="_tmp_openai_api_key",
                value=st.session_state.openai_api_key,
                on_change=lambda: setattr(
                    st.session_state, 
                    "openai_api_key", 
                    st.session_state._tmp_openai_api_key
                )
            )

    st.caption("Evaluation")
    with st.expander("Evaluation", expanded=False):
        st.checkbox("Run RAGAS", disabled=True)

# MAIN AREA --------------------------------------------------------------------
col_chat, col_debug = st.columns([3, 2])

# CHAT -------------------------------------------------------------------------
with col_chat:
    st.header("Chat")
    chat_container = st.container(height=600)

    # render messages
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    for i, chunk in enumerate(msg.get("chunks", []), 1):
                        score = chunk.get("score", 0)
                        score_color = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
                        
                        with st.expander(f"{score_color} Score: {score:.2f} | 📍 Page: {chunk['page']}"):
                            st.text(chunk["text"])
                    if msg.get("generation_enabled") and msg.get("content"):
                        st.markdown(msg["content"])
                    else:
                        st.info("Generation disabled or no api key, showing retrieved chunks only.")
                else:
                    st.markdown(msg["content"])

    # input
    if not st.session_state.collection:
        st.chat_input("Upload and index a PDF first...", disabled=True)
    else:
        if question := st.chat_input("Ask anything about your document..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.spinner("Retrieving and generating..."):
                # query handling
                llm_fn = quick_llm(st.session_state.provider, st.session_state.model,
                                st.session_state.anthropic_api_key if st.session_state.provider == "claude" 
                                else st.session_state.openai_api_key)
                
                query_result = handle_query(question, st.session_state.query_strategy, llm_fn)
                
                # retrieve for each query, merge results
                all_chunks = {}
                query_chunk_map = {}  # sub_question → [chunk_ids]

                for q in query_result["queries"]:
                    found = retrieve(q, st.session_state.collection,
                                    st.session_state.retrieval_n_chunks,
                                    st.session_state.indexer,
                                    st.session_state.retrieval_model)
                    query_chunk_map[q] = []
                    for chunk in found:
                        all_chunks[chunk["chunk_id"]] = chunk # deduplicate by id
                        query_chunk_map[q].append(chunk["chunk_id"])
                chunks_found = list(all_chunks.values())[:st.session_state.retrieval_n_chunks]
            if st.session_state.enable_generation and has_api_key:
                with st.spinner("Generating..."):
                    answer = generate(question, chunks_found, st.session_state.provider, st.session_state.model, st.session_state.max_tokens, (st.session_state.anthropic_api_key if st.session_state.provider == "claude" else st.session_state.openai_api_key))
            else:
                answer = None
            pages_used = sorted(set(c["page"] for c in chunks_found))
            st.caption(f"Sources: pages {pages_used}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "chunks": chunks_found,
                "generation_enabled": st.session_state.enable_generation and has_api_key,
            })

            # store debug info
            st.session_state.last_debug = {
                "question": question,
                "query_used": query_result["queries"][0],
                "query_strategy": query_result["strategy"],
                "query_display": query_result["display"],
                "chunks": chunks_found,
                "query_chunk_map": query_chunk_map,
            }

            for key in list(st.session_state.keys()):
                if key.startswith("chunk_display_") or key.startswith("child_") or key.startswith("parent_"):
                    del st.session_state[key]
            st.rerun()

# DEBUG PANE ---------------------------------------------------------------
with col_debug:
    st.header("🔬 Debug Panel")

    if st.session_state.last_debug is None:
        st.info("Ask a question to see debug info here.")
    else:
        debug = st.session_state.last_debug

        # costs
        if st.session_state.retrieval_model == "OpenAI Embedding":
            if hasattr(st.session_state.indexer, "tokens_used"):
                with st.expander("💵 Embedding tokens used", expanded=True):
                    st.markdown(f"total: :green[{sum(st.session_state.indexer.tokens_used.values())}]")
                    for event, tokens_used in st.session_state.indexer.tokens_used.items():
                        st.markdown(f"{event}: :green[{tokens_used:,}]")

        # query trace
        with st.expander("💬 Query trace", expanded=False):
            st.markdown(f"**Strategy:** `{debug['query_strategy']}`")
            st.markdown(f"**Original:** {debug['question']}")
            
            display = debug.get("query_display", {})
            
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
                score = chunk.get("score", 0)
                if score >= 0.7:
                    score_color = "🟢"
                elif score >= 0.4:
                    score_color = "🟡"
                else:
                    score_color = "🔴"

                st.markdown(
                    f"**Chunk {i}** — Page {chunk['page']} "
                    f"{score_color} score `{score}`"
                )

                if debug.get("query_chunk_map"):
                    matched_qs = [q for q, ids in debug["query_chunk_map"].items() 
                                if chunk["chunk_id"] in ids]
                    if matched_qs:
                        for q in matched_qs:
                            st.caption(f"↳ from: {q}")

                if chunk.get("parent_text"):
                    tab_child, tab_parent = st.tabs(["Matched (child)", "Sent to LLM (parent)"])
                    with tab_child:
                        st.text_area("Matched (child)", chunk["text"], height=100, disabled=True, key=f"child_{i}", label_visibility="collapsed")
                    with tab_parent:
                        st.text_area("Sent to LLM (parent)", chunk["parent_text"], height=100, disabled=True, key=f"parent_{i}", label_visibility="collapsed")
                else:
                    st.text_area("Chunk", chunk["text"], height=100, disabled=True, label_visibility="collapsed")

        # process log
        with st.expander("🛠️ Process log", expanded=False):
            st.markdown("**Steps run:**")
            st.markdown("1. ✅ Query received")
            st.markdown("2. ✅ Query strategy: none (passthrough)")
            st.markdown("3. ✅ Dense retrieval")
            st.markdown("4. ⬜ Reranker (disabled)")
            st.markdown("5. ⬜ Corrective RAG (disabled)")
            st.markdown("6. ⬜ Contextual compression (disabled)")
            st.markdown("7. ✅ Generation")
            st.markdown("8. ⬜ RAGAS evaluation (disabled)")
# debug state if needed
st.write({k: v for k, v in dict(st.session_state).items() if k not in ('anthropic_api_key','openai_api_key','_tmp_anthropic_api_key','_tmp_openai_api_key')})