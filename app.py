import os
import streamlit as st
from langchain_openai import ChatOpenAI
from backend import fetch_arxiv_papers, build_survey_text

# Extra dependencies
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from docx import Document


# Support Streamlit secrets or env var for OpenAI key
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


def export_to_pdf(text: str) -> BytesIO:
    """Convert text into a downloadable PDF."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    flow = []

    for line in text.split("\n"):
        if line.strip():
            flow.append(Paragraph(line.strip(), styles["Normal"]))
            flow.append(Spacer(1, 8))

    doc.build(flow)
    buffer.seek(0)
    return buffer


def export_to_word(text: str) -> BytesIO:
    """Convert text into a downloadable Word docx."""
    buffer = BytesIO()
    doc = Document()

    for line in text.split("\n"):
        if line.strip():
            doc.add_paragraph(line.strip())

    doc.save(buffer)
    buffer.seek(0)
    return buffer


def main():
    st.set_page_config(page_title="Research Assistant", page_icon="📚", layout="wide")
    st.title("📚 RAG Based Research Assistant")
    st.write("Fetch papers and generate a structured literature survey.")

    # Sidebar
    with st.sidebar:
        topic = st.text_input("Research Topic", "Blue light effect on lettuce")
        n_fetch = st.slider("Fetch papers from arXiv", 5, 50, 15, 5)
        top_k = st.slider("Top-K for summarization", 3, 10, 5, 1)
        model_name = st.selectbox("LLM", ["gpt-4", "gpt-3.5-turbo"], index=1)
        sort_by = st.selectbox("arXiv sorting", ["relevance", "submittedDate"], index=0)
        generate = st.button("🔎 Generate Survey")

    if not generate:
        return

    # Fetch papers
    with st.spinner("Fetching papers..."):
        papers = fetch_arxiv_papers(topic, max_results=n_fetch, sort_by=sort_by)

    if not papers:
        st.error("No papers found.")
        return

    st.success(f"Retrieved {len(papers)} papers.")
    with st.expander("📄 Retrieved papers"):
        for d in papers:
            st.markdown(
                f"- **{d.metadata['title']}** ({d.metadata['year']}) — "
                f"[{d.metadata['link']}]({d.metadata['link']})"
            )

    # GPT summarization
    with st.spinner("Generating survey..."):
        llm = ChatOpenAI(model=model_name, temperature=0.3)
        survey = build_survey_text(topic, papers[:top_k], llm)

    st.subheader("📑 Literature Survey Draft")
    st.write(survey)

    # Export buttons
    st.subheader("📤 Export Report")
    col1, col2 = st.columns(2)

    with col1:
        pdf_file = export_to_pdf(survey)
        st.download_button(
            "⬇️ Download PDF",
            data=pdf_file,
            file_name="literature_survey.pdf",
            mime="application/pdf",
        )

    with col2:
        word_file = export_to_word(survey)
        st.download_button(
            "⬇️ Download Word",
            data=word_file,
            file_name="literature_survey.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )


if __name__ == "__main__":
    main()
