# main.py

import os
import re
import time
from html import escape
from io import StringIO, BytesIO

import streamlit as st
from docx import Document
from langchain.embeddings import OpenAIEmbeddings

# from mysecrets import OPENAI_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

from utils import (
    chunk_embed_store_transcript,
    build_retriever,
    get_llm_client,
    generate_insights,
    split_insights_into_points,
    find_supporting_quotes,
    export_to_word,
    extract_insight_summaries,
)


def parse_transcript(uploaded_file):
    """Reads txt or docx using python-docx (no docx2txt dependency)."""
    if uploaded_file.type == "text/plain":
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore"))
        return stringio.read()

    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        file_bytes = BytesIO(uploaded_file.getvalue())
        doc = Document(file_bytes)
        return "\n".join(p.text for p in doc.paragraphs)

    return ""


def main():
    # Title + tagline
    st.markdown("<h1 style='font-weight:bold;'>💡Decode</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:gray; font-size:14px; margin-top:-10px;'>Qualitative insights you can trace, and trust</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Transcript upload
    st.subheader("Upload raw data - interview transcript(s)")
    uploaded_files = st.file_uploader(
        "You may upload more than one", type=["docx", "txt"], accept_multiple_files=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if uploaded_files:
        all_transcripts = [parse_transcript(f) for f in uploaded_files]

        # --- Research question: form submit + validation ---
        st.markdown("<h3 style='font-weight:bold;'>Enter Your Research Question</h3>", unsafe_allow_html=True)
        placeholder_text = (
            "What are you trying to find out from this transcript? For example, "
            "understand user perceptions towards the new mobile app."
        )
        with st.form(key="rq_form", clear_on_submit=False):
            research_question = st.text_area("", placeholder=placeholder_text, height=80, key="rq_text")
            submitted = st.form_submit_button("Submit Research Question", type="primary")

        if submitted:
            if not research_question or not research_question.strip():
                st.error("Please enter a research question before submitting.")
                st.stop()
            st.session_state["research_question"] = research_question.strip()

        if "research_question" in st.session_state:
            rq = st.session_state["research_question"]
            # Styled single display of the RQ (avoid duplication below)
            st.markdown(
                f"""
                <div style="color:#6b7280; font-size:14px; margin:6px 0 14px 0;">
                    <span style="text-transform:uppercase; letter-spacing:.04em; font-weight:600; color:#374151;">
                        Research question
                    </span><br>
                    <em>{escape(rq)}</em>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # --- Build vector store once per session ---
            if "vectordb" not in st.session_state:
                with st.spinner("Processing transcripts and generating embeddings..."):
                    all_chunks = []
                    all_chunk_sources = []  # transcript index per chunk
                    for idx, transcript in enumerate(all_transcripts):
                        text_splitter = chunk_embed_store_transcript.__globals__['RecursiveCharacterTextSplitter'](
                            separators=["\n\n", "\n", " ", ""],
                            chunk_size=1000,
                            chunk_overlap=100,
                        )
                        chunks = text_splitter.split_text(transcript)
                        all_chunks.extend(chunks)
                        all_chunk_sources.extend([idx] * len(chunks))
                        time.sleep(0.05)

                    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

                    from langchain.vectorstores import Chroma
                    vectordb = Chroma.from_texts(
                        texts=all_chunks,
                        embedding=embeddings_model,
                        persist_directory="./chroma_db",
                        collection_name="decode_transcripts",
                    )
                    vectordb.persist()
                    st.session_state["vectordb"] = vectordb
                    st.session_state["all_chunks"] = all_chunks
                    st.session_state["embeddings_model"] = embeddings_model
                    st.session_state["all_chunk_sources"] = all_chunk_sources
                    st.session_state["all_transcripts"] = all_transcripts

            all_chunks = st.session_state["all_chunks"]
            embeddings_model = st.session_state["embeddings_model"]
            all_chunk_sources = st.session_state["all_chunk_sources"]

            retriever = build_retriever()
            client = get_llm_client()

            relevant_docs = retriever.get_relevant_documents(rq)

            with st.spinner("Generating insights, please wait..."):
                insight_text = generate_insights(client, rq, relevant_docs)
                insight_points = split_insights_into_points(insight_text)
                summaries = extract_insight_summaries(insight_points)
                supporting_quotes = find_supporting_quotes(insight_points, all_chunks, embeddings_model)

            # --- Findings ---
            st.markdown("### Findings:", unsafe_allow_html=True)

            for i, point in enumerate(insight_points):
                summary = summaries[i].strip(" *")

                # Remove summary from body even if bolded/punctuated
                summary_pattern = re.compile(r'^(\*\*)?' + re.escape(summary) + r'(\*\*)?[\s:.\-–—]*', re.IGNORECASE)
                body = summary_pattern.sub('', point, count=1).lstrip('\n :.-–—')

                # Map quotes back to transcript indices for participant count
                participant_indices = set()
                for quote in supporting_quotes[i]:
                    try:
                        chunk_idx = all_chunks.index(quote)
                        participant_indices.add(all_chunk_sources[chunk_idx])
                    except ValueError:
                        pass

                with st.expander(f"**Insight {i+1}: {summary}**"):
                    # Subtle linkage to the research question
                    st.markdown(
                        f"<div style='color:#6b7280; font-size:13px; margin:-2px 0 10px 0;'>"
                        f"In response to: <em>{escape(rq)}</em>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Body (no repeated headline)
                    st.write(body)

                    # Mentioned by
                    st.markdown(
                        f"<span style='color: gray; font-size: 13px;'>"
                        f"Mentioned by: {len(participant_indices)} participant{'s' if len(participant_indices) != 1 else ''}"
                        f"</span>",
                        unsafe_allow_html=True,
                    )

                    # Supporting quotes (clean rendering)
                    st.markdown("**Supporting quotes:**")

                    def render_quote(raw_quote: str):
                        q = raw_quote.strip()
                        lines = q.splitlines()
                        timestamp = None
                        text = q

                        # If first line is a standalone timestamp
                        if lines and re.match(r'^\d{2}:\d{2}:\d{2}$', lines[0].strip()):
                            timestamp = lines[0].strip()
                            text = "\n".join(lines[1:]).strip()
                        else:
                            m = re.match(r'^(\d{2}:\d{2}:\d{2})\s*(.*)$', q)
                            if m:
                                timestamp = m.group(1)
                                text = m.group(2).strip()

                        # Remove speaker labels
                        text = re.sub(r'^(Speaker\s*\d+\s*:?)\s*', '', text, flags=re.IGNORECASE)

                        # Fallback excerpt if empty
                        if not text:
                            fallback = "\n".join(lines[1:]).strip() if lines else q
                            text = (fallback[:220] + "…") if len(fallback) > 220 else fallback

                        if timestamp:
                            st.markdown(
                                f"<span style='font-size: 13px; color: #333;'>"
                                f"<span style='font-weight:bold;'>{timestamp}</span><br>"
                                f"<i style='font-size: 12px; white-space: pre-wrap;'>{escape(text)}</i>"
                                f"</span>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"<span style='font-size: 13px; color: #333;'>"
                                f"<i style='font-size: 12px; white-space: pre-wrap;'>{escape(text)}</i>"
                                f"</span>",
                                unsafe_allow_html=True,
                            )

                    # De-dup quotes
                    seen = set()
                    for quote in supporting_quotes[i]:
                        if quote in seen:
                            continue
                        seen.add(quote)
                        render_quote(quote)

            st.markdown("---")
            st.markdown("## 📤 Export Findings")

            if st.button("Download Findings as Word Document"):
                doc_stream = export_to_word([(rq, insight_text)], [supporting_quotes])
                st.download_button(
                    label="📥 Click to Download",
                    data=doc_stream,
                    file_name="decode_findings.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )


if __name__ == "__main__":
    main()
