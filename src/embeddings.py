from langchain_huggingface import HuggingFaceEmbeddings


# Registry of supported embedding models with metadata
EMBEDDING_REGISTRY = {
    "all-MiniLM-L6-v2": {
        "dim": 384,
        "provider": "huggingface",
        "description": "Lightweight, fast 384-dim sentence embeddings",
    },
    "BAAI/bge-small-en-v1.5": {
        "dim": 384,
        "provider": "huggingface",
        "description": "BGE small 384-dim model optimized for retrieval",
    },
    "BAAI/bge-large-en-v1.5": {
        "dim": 1024,
        "provider": "huggingface",
        "description": "BGE large 1024-dim model for higher accuracy retrieval",
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "dim": 768,
        "provider": "huggingface",
        "description": "MPNet base 768-dim strong general-purpose model",
    },
}


def get_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    """
    Return a LangChain-compatible embedding model by name.
    """
    if model_name in EMBEDDING_REGISTRY:
        info = EMBEDDING_REGISTRY[model_name]
        print(
            f"Embedding model: {model_name}\n"
            f"  Dimension  : {info['dim']}\n"
            f"  Description: {info['description']}"
        )
    else:
        print(f"Embedding model: {model_name} (custom / not in registry)")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def list_models():
    """Print all registered embedding models."""
    print("\nAvailable Embedding Models:")
    for name, info in EMBEDDING_REGISTRY.items():
        print(f"  - {name:50s} dim={info['dim']:5d}  {info['description']}")
