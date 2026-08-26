from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import fitz
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"


@dataclass
class SourceChunk:
    rank: int
    text: str
    distance: float
    page: int | None = None


@dataclass
class PaperIndex:
    title: str
    chunks: list[dict[str, Any]]
    index: Any


def get_api_key() -> str:
    try:
        import streamlit as st

        secret = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret = ""
    return secret or os.getenv("GROQ_API_KEY", "")


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def extract_pages(pdf_bytes: bytes) -> list[dict[str, Any]]:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    return [
        {"page": page_number, "text": page.get_text("text")}
        for page_number, page in enumerate(document, start=1)
    ]


def chunk_pages(
    pages: list[dict[str, Any]], chunk_size: int = 1000, overlap: int = 200
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for page_data in pages:
        text = " ".join(page_data["text"].split())
        if not text:
            continue
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append({"text": text[start:end], "page": page_data["page"]})
            if end >= len(text):
                break
            start = end - overlap
    return chunks


def build_index(chunks: list[dict[str, Any]], model: SentenceTransformer) -> Any:
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def build_paper_index(
    pdf_bytes: bytes, filename: str, model: SentenceTransformer
) -> PaperIndex:
    pages = extract_pages(pdf_bytes)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No selectable text was found in this PDF.")
    return PaperIndex(Path(filename).stem, chunks, build_index(chunks, model))


def retrieve(
    paper: PaperIndex, query: str, model: SentenceTransformer, top_k: int = 5
) -> list[SourceChunk]:
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = paper.index.search(query_embedding, min(top_k, paper.index.ntotal))
    return [
        SourceChunk(
            rank=rank,
            text=paper.chunks[index]["text"],
            distance=float(distance),
            page=paper.chunks[index].get("page"),
        )
        for rank, (distance, index) in enumerate(
            zip(distances[0], indices[0]), start=1
        )
        if index >= 0
    ]


def generate_answer(
    paper: PaperIndex,
    query: str,
    model: SentenceTransformer,
    top_k: int = 5,
) -> tuple[str, list[SourceChunk]]:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    sources = retrieve(paper, query, model, top_k)
    context = "\n\n".join(
        f"[Source {source.rank}] {source.text}" for source in sources
    )
    prompt = f"""You are a careful research assistant. Answer the question using ONLY the supplied paper excerpts.
If the excerpts do not contain enough information, say exactly that you could not find a supported answer in this paper. Do not use outside knowledge. Keep the answer concise and distinguish reported findings from interpretation.

Paper: {paper.title}
Excerpts:
{context}

Question: {query}

Answer:"""
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content or "No answer was returned.", sources
