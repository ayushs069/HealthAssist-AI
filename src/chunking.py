from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter


# Predefined chunk size configurations for comparison
CHUNK_CONFIGS = {
    "small": {"chunk_size": 300, "chunk_overlap": 30},
    "medium": {"chunk_size": 500, "chunk_overlap": 50},
    "large": {"chunk_size": 1000, "chunk_overlap": 100},
}


def split_documents(documents, chunk_size=500, chunk_overlap=50, strategy="recursive"):
    """
    Split documents using configurable chunking strategies.

    Args:
        documents     : List of LangChain Document objects
        chunk_size    : Target size of each chunk (characters or tokens)
        chunk_overlap : Overlap between consecutive chunks
        strategy      : "recursive" | "token"
    """
    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    elif strategy == "token":
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. Choose 'recursive' or 'token'."
        )

    chunks = splitter.split_documents(documents)
    print(
        f"Chunking complete: strategy={strategy} | "
        f"size={chunk_size} | overlap={chunk_overlap} | chunks={len(chunks)}"
    )
    return chunks
