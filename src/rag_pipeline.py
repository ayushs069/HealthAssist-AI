import os
import sys
from typing import Any, Iterator, List, Tuple

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # dotenv is optional; environment variables can still be injected externally.
    pass

from langchain_core.documents import Document

from loader import load_pdf
from chunking import split_documents
from embeddings import get_embeddings
from vectorstores import create_vector_store, retrieve_documents
from llm import get_llm
from memory_utils import warn_if_low_memory
from prompts import build_prompt


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def build_pipeline(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    chunk_strategy: str = "recursive",
    embedding_model: str = "all-MiniLM-L6-v2",
    db_type: str = "faiss",
    persist_dir: str = None,
) -> object:
    """
    Build the full RAG ingestion pipeline.
    """
    print("\n" + ("=" * 60))
    print("Building RAG pipeline")
    print("=" * 60)
    print(f"PDF file        : {os.path.basename(pdf_path)}")
    print(f"Chunk strategy  : {chunk_strategy} | size={chunk_size} | overlap={chunk_overlap}")
    print(f"Embedding model : {embedding_model}")
    print(f"Vector DB       : {db_type.upper()}")
    memory_warning_gb = _get_env_float("MEMORY_WARNING_GB", 2.0)
    available_gb = warn_if_low_memory(
        min_available_gb=memory_warning_gb,
        context="RAG pipeline build",
    )
    if available_gb is not None:
        print(f"Available RAM   : {available_gb:.2f} GB")

    print("\n[1/4] Loading and preprocessing documents...")
    docs = load_pdf(pdf_path)

    print("\n[2/4] Chunking documents...")
    chunks = split_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=chunk_strategy,
    )

    print(f"\n[3/4] Generating embeddings ({embedding_model})...")
    embeddings = get_embeddings(embedding_model)

    print(f"\n[4/4] Building {db_type.upper()} vector store...")
    vectorstore = create_vector_store(
        chunks, embeddings, db_type=db_type, persist_dir=persist_dir
    )

    print(f"\nPipeline ready ({len(chunks)} chunks indexed)\n")
    return vectorstore


def generate_answer(
    vectorstore: Any,
    query: str,
    retrieval_query: str = None,
    llm_model: str = "ollama:llama3",
    retrieval_type: str = "similarity",
    prompt_template: str = "basic",
    k: int = 3,
    llm_timeout_seconds: int = 80,
) -> Tuple[str, List[Document], str]:
    """
    Generate an answer using Retrieval-Augmented Generation.

    Returns:
        Tuple[str, List[Document], str]:
            answer_text, retrieved_docs, prompt
    """
    docs, prompt = _prepare_prompt_and_docs(
        vectorstore=vectorstore,
        query=query,
        retrieval_query=retrieval_query,
        retrieval_type=retrieval_type,
        prompt_template=prompt_template,
        k=k,
    )
    llm = get_llm(llm_model, timeout_seconds=llm_timeout_seconds)
    response = llm.invoke(prompt)
    answer_text = response.content if hasattr(response, "content") else str(response)
    return answer_text, docs, prompt


def generate_answer_stream(
    vectorstore: Any,
    query: str,
    retrieval_query: str = None,
    llm_model: str = "ollama:llama3",
    retrieval_type: str = "similarity",
    prompt_template: str = "basic",
    k: int = 3,
    llm_timeout_seconds: int = 80,
) -> Tuple[Iterator[str], List[Document], str]:
    """
    Stream an answer token-by-token for responsive UI rendering.

    Returns:
        Tuple[Iterator[str], List[Document], str]:
            stream_iterator, retrieved_docs, prompt
    """
    docs, prompt = _prepare_prompt_and_docs(
        vectorstore=vectorstore,
        query=query,
        retrieval_query=retrieval_query,
        retrieval_type=retrieval_type,
        prompt_template=prompt_template,
        k=k,
    )
    llm = get_llm(llm_model, timeout_seconds=llm_timeout_seconds)

    if hasattr(llm, "stream"):
        stream_kwargs = {"stream": True} if llm_model.startswith("ollama:") else {}

        def token_stream() -> Iterator[str]:
            for chunk in llm.stream(prompt, **stream_kwargs):
                if hasattr(chunk, "content"):
                    text = chunk.content
                else:
                    text = str(chunk)
                if text:
                    yield text

        return token_stream(), docs, prompt

    def fallback_stream() -> Iterator[str]:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        if text:
            yield text

    return fallback_stream(), docs, prompt


def _prepare_prompt_and_docs(
    vectorstore: Any,
    query: str,
    retrieval_query: str = None,
    retrieval_type: str = "similarity",
    prompt_template: str = "basic",
    k: int = 3,
) -> Tuple[List[Document], str]:
    """
    Retrieve top-k documents and construct the final prompt context.
    """
    effective_retrieval_query = retrieval_query or query
    docs_with_scores = retrieve_documents(
        vectorstore, effective_retrieval_query, k=k, retrieval_type=retrieval_type
    )

    formatted_docs = []
    for item in docs_with_scores:
        if isinstance(item, tuple):
            formatted_docs.append(item)
        else:
            formatted_docs.append((item, 0.0))

    context_parts = []
    for i, (doc, score) in enumerate(formatted_docs):
        page = doc.metadata.get("page", "?")
        source = os.path.basename(doc.metadata.get("source", "document"))
        doc.metadata["retrieval_score"] = float(score)
        context_parts.append(
            f"[Chunk {i + 1} | Source: {source} | Page: {page}]\n{doc.page_content}"
        )
    context = "\n\n".join(context_parts)
    prompt = build_prompt(context, query, template_name=prompt_template)
    docs = [d for d, _ in formatted_docs]
    return docs, prompt
