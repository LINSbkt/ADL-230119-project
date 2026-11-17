#!/usr/bin/env python3
"""Task 4: Streamlit UI for the Integrated Sciences handbook chatbot."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.config import Settings
from rag.pipeline import RAGPipeline


@st.cache_resource(show_spinner="Loading handbook index and Ollama models...")
def get_pipeline(llm_model: str) -> RAGPipeline:
    settings = Settings.from_env(
        data_dir=ROOT / "data" / "pdfs",
        vector_store=ROOT / "data" / "pdfs" / "vector_store",
        collection="is_handbook",
        strict_context=True,
        llm_model=llm_model,
    )
    return RAGPipeline(settings, rebuild=False)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.caption("No retrieved sources.")
        return
    st.markdown("**Sources**")
    for source in sources:
        page = source.get("page")
        page_text = f", page {page}" if page not in (None, "") else ""
        with st.expander(f"{source['source']}{page_text} · score {source['score']}"):
            st.write(source["content"])


def main() -> None:
    st.set_page_config(page_title="IS Handbook Chatbot", page_icon="📘", layout="wide")
    st.title("Integrated Sciences Handbook Chatbot")
    st.caption("Local RAG assistant for Fulbright's B.S. Integrated Sciences major handbook.")

    with st.sidebar:
        st.header("Settings")
        llm_model = st.text_input("Ollama LLM", value="qwen2.5:0.5b")
        st.caption("Knowledge source: `IS_MajorHandbook.pdf`")
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    pipeline = get_pipeline(llm_model)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Ask about IS requirements, courses, outcomes, or policies")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving handbook passages and generating an answer..."):
            result = pipeline.ask(question)
        st.markdown(result["answer"])
        if result["used_fallback"]:
            st.info("No sufficiently relevant handbook chunks were retrieved.")
        render_sources(result["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )


if __name__ == "__main__":
    main()
