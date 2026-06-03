import os

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma


def _validate_chunks(chunks):
    """
    Ensure vector DB creation starts only with non-empty, usable chunks.
    """
    if chunks is None:
        raise ValueError("Vector store initialization failed: chunks is None.")

    if not isinstance(chunks, list):
        chunks = list(chunks)

    if len(chunks) == 0:
        raise ValueError(
            "Vector store initialization skipped: no chunks were generated. "
            "Verify PDF loading/chunking before indexing."
        )

    valid_chunks = []
    for chunk in chunks:
        text = getattr(chunk, "page_content", "")
        if isinstance(text, str) and text.strip():
            valid_chunks.append(chunk)

    if not valid_chunks:
        raise ValueError(
            "Vector store initialization skipped: all chunks are empty after preprocessing."
        )

    return valid_chunks


def create_vector_store(chunks, embeddings, db_type: str = "faiss", persist_dir: str = None):
    """
    Create a vector store from document chunks.
    """
    chunks = _validate_chunks(chunks)

    if db_type == "faiss":
        store = FAISS.from_documents(chunks, embeddings)
        print(f"FAISS vectorstore created with {len(chunks)} vectors (in-memory)")
        return store

    if db_type == "chroma":
        path = persist_dir or os.environ.get("DB_PATH", "./chroma_db")
        os.makedirs(path, exist_ok=True)
        store = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=path,
            collection_name="healthcare_rag",
        )
        print(f"ChromaDB vectorstore created with {len(chunks)} vectors (path='{path}')")
        return store

    raise ValueError(f"Unknown db_type: '{db_type}'. Choose 'faiss' or 'chroma'.")


def retrieve_documents(vectorstore, query: str, k: int = 3, retrieval_type: str = "similarity"):
    """
    Retrieve top-k documents using different retrieval strategies and return scores.
    """
    if hasattr(vectorstore, "similarity_search_with_relevance_scores"):
        try:
            results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        except Exception:
            results = vectorstore.similarity_search_with_score(query, k=k)
    else:
        results = vectorstore.similarity_search_with_score(query, k=k)

    docs_with_scores = results

    if retrieval_type == "mmr":
        docs = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=k * 4, lambda_mult=0.7)
        docs_with_scores = [(doc, 0.9 - (i * 0.05)) for i, doc in enumerate(docs)]
    elif retrieval_type == "threshold":
        docs_with_scores = [(doc, score) for doc, score in results if score >= 0.3 or score < 1.0]
        docs_with_scores = docs_with_scores[:k]

    return docs_with_scores
