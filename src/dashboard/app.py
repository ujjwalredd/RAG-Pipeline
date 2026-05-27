import streamlit as st
import httpx
import json

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="RAG Pipeline", page_icon="🔍", layout="wide")
st.title("RAG Pipeline — Hybrid Search Dashboard")


# --- Sidebar ---
with st.sidebar:
    st.header("Settings")

    retrieval_mode = st.selectbox("Retrieval Mode", ["hybrid", "dense", "sparse"])
    use_reranker = st.checkbox("Use Reranker", value=True)
    dense_weight = st.slider("Dense Weight", 0.0, 1.0, 0.7, 0.05)
    sparse_weight = st.slider("Sparse Weight", 0.0, 1.0, 0.3, 0.05)

    st.divider()
    st.header("System Status")

    try:
        health = httpx.get(f"{API_BASE}/health", timeout=5.0).json()
        st.metric("Indexed Chunks", health["indexed_chunks"])
        st.metric("Ollama", "Online" if health["ollama_available"] else "Offline")
    except httpx.HTTPError:
        st.error("API not reachable. Start with: uvicorn src.api.main:app --port 8000")

    st.divider()
    st.header("Ingest Documents")
    with st.form("ingest_form"):
        max_docs = st.number_input("Max Documents", min_value=1, max_value=100000, value=100)
        strategy = st.selectbox("Chunking Strategy", ["recursive", "fixed_size", "semantic"])
        ingest_btn = st.form_submit_button("Ingest")

    if ingest_btn:
        with st.spinner("Ingesting documents..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/v1/ingest",
                    json={"max_docs": max_docs, "chunking_strategy": strategy},
                    timeout=600.0,
                )
                resp.raise_for_status()
                data = resp.json()
                st.success(
                    f"Ingested {data['documents']} docs → "
                    f"{data['chunks_indexed']} chunks ({data['strategy']})"
                )
            except httpx.HTTPError as e:
                st.error(f"Ingest failed: {e}")


# --- Main Area ---
question = st.text_input("Ask a question about the documentation:", placeholder="e.g., What are the default file upload limits?")

col_ask, col_compare = st.columns([1, 1])
ask_btn = col_ask.button("Ask", type="primary", use_container_width=True)
compare_btn = col_compare.button("Compare Hybrid vs Dense", use_container_width=True)

if ask_btn and question:
    with st.spinner("Searching and generating..."):
        try:
            resp = httpx.post(
                f"{API_BASE}/v1/ask",
                json={
                    "question": question,
                    "use_reranker": use_reranker,
                    "dense_weight": dense_weight,
                    "sparse_weight": sparse_weight,
                    "retrieval_mode": retrieval_mode,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            st.error(f"Request failed: {e}")
            st.stop()

    # Answer
    st.subheader("Answer")
    if not data["is_confident"]:
        st.warning("Low confidence — answer may be incomplete or unreliable")
    st.markdown(data["answer"])

    # Confidence
    st.subheader("Confidence Scores")
    conf = data["confidence"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Composite", f"{conf['composite']:.1%}")
    c2.metric("Retrieval", f"{conf['retrieval_confidence']:.1%}")
    c3.metric("Citations", f"{conf['citation_coverage']:.1%}")
    c4.metric("Completeness", f"{conf['answer_completeness']:.1%}")

    # Citations
    if data["citations"]:
        st.subheader("Citation Verification")
        for cit in data["citations"]:
            status = "✅" if cit["supported"] else "❌"
            st.markdown(f"{status} **[{cit['citation_id']}]** {cit['claim']}")

    # Retrieved Chunks
    st.subheader("Retrieved Chunks")
    for i, chunk in enumerate(data["context_chunks"], 1):
        with st.expander(
            f"[{i}] {chunk['metadata'].get('title', 'Unknown')} — "
            f"Score: {chunk['score']:.4f} | "
            f"Type: {chunk['metadata'].get('source_type', 'N/A')}"
        ):
            st.text(chunk["content"][:1000])
            st.json(chunk["metadata"])


if compare_btn and question:
    st.subheader("Hybrid vs Dense-Only Comparison")

    col_hybrid, col_dense = st.columns(2)

    with col_hybrid:
        st.markdown("### Hybrid Retrieval")
        with st.spinner("Running hybrid..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/v1/ask",
                    json={
                        "question": question,
                        "retrieval_mode": "hybrid",
                        "use_reranker": use_reranker,
                        "dense_weight": dense_weight,
                        "sparse_weight": sparse_weight,
                    },
                    timeout=120.0,
                )
                resp.raise_for_status()
                hybrid_data = resp.json()
                st.markdown(hybrid_data["answer"])
                st.metric("Confidence", f"{hybrid_data['confidence']['composite']:.1%}")
                st.metric("Chunks Retrieved", len(hybrid_data["context_chunks"]))
            except httpx.HTTPError as e:
                st.error(f"Hybrid failed: {e}")

    with col_dense:
        st.markdown("### Dense-Only Retrieval")
        with st.spinner("Running dense-only..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/v1/ask",
                    json={
                        "question": question,
                        "retrieval_mode": "dense",
                        "use_reranker": False,
                        "dense_weight": 1.0,
                        "sparse_weight": 0.0,
                    },
                    timeout=120.0,
                )
                resp.raise_for_status()
                dense_data = resp.json()
                st.markdown(dense_data["answer"])
                st.metric("Confidence", f"{dense_data['confidence']['composite']:.1%}")
                st.metric("Chunks Retrieved", len(dense_data["context_chunks"]))
            except httpx.HTTPError as e:
                st.error(f"Dense failed: {e}")
