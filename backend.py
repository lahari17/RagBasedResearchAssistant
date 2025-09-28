import os
import requests
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.docstore.document import Document


# --------------------------
# 1) Fetch papers from arXiv
# --------------------------
def fetch_arxiv_papers(query: str, max_results: int = 20, sort_by: str = "relevance"):
    """
    Fetch papers from arXiv API and parse results into Document objects.
    """
    base = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }
    response = requests.get(base, params=params, timeout=30)
    text = response.text

    entries = text.split("<entry>")[1:]
    papers = []

    for entry in entries:
        try:
            title = entry.split("<title>")[1].split("</title>")[0].strip()
            summary = entry.split("<summary>")[1].split("</summary>")[0].strip()
            link = entry.split("<id>")[1].split("</id>")[0].strip()
            published = entry.split("<published>")[1].split("</published>")[0].strip()
            year = int(published.split("-")[0])

            authors = []
            for part in entry.split("<author>")[1:]:
                name = part.split("<name>")[1].split("</name>")[0].strip()
                authors.append(name)

            papers.append(
                Document(
                    page_content=summary,
                    metadata={"title": title, "link": link, "year": year, "authors": authors},
                )
            )
        except Exception:
            continue

    return papers


# --------------------------
# 2) GPT Literature Survey Summarization
# --------------------------
def build_survey_text(query: str, docs, llm: ChatOpenAI):
    """
    Use GPT to summarize retrieved papers into a structured survey.
    """
    template = """
You are an assistant that writes **professional literature survey drafts**.

Organize the information into the following sections:
1. **Background** → Provide a concise overview of the research area.
2. **Key Themes** → Summarize recurring ideas/findings as bullet points.
3. **Research Gaps** → Highlight missing, underexplored, or inconsistent areas.
4. **References** → List titles with year (from the provided metadata). Do not invent references.

Papers:
{papers}

Research Topic: {query}
"""
    prompt = PromptTemplate(template=template, input_variables=["papers", "query"])
    chain = prompt | llm

    papers_text = "\n\n".join(
        [f"Title: {d.metadata.get('title','(no title)')} ({d.metadata.get('year','')})\nSummary: {d.page_content}" for d in docs]
    )
    result = chain.invoke({"papers": papers_text, "query": query})
    return result.content
