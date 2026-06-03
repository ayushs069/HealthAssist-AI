import os
import re
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str) -> list:
    """
    Load PDF and perform text preprocessing.

    Steps:
    - Load all pages using PyPDFLoader
    - Clean whitespace and noise
    - Filter out near-empty pages (<50 chars)
    - Preserve page metadata (page number, source)

    Returns:
        List of LangChain Document objects
    """
    if not file_path:
        raise ValueError("PDF path is empty. Provide a valid PDF file path.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: '{file_path}'")

    loader = PyPDFLoader(file_path)
    try:
        raw_docs = loader.load()
    except Exception as exc:
        error_name = type(exc).__name__
        error_text = str(exc)
        lower_text = error_text.lower()

        if (
            error_name in {"PdfStreamError", "PdfReadError"}
            or "invalid pdf header" in lower_text
            or "eof marker not found" in lower_text
        ):
            raise ValueError(
                f"Unable to read '{file_path}'. The PDF appears corrupt or uses an unsupported binary stream."
            ) from exc

        raise RuntimeError(f"Failed to load PDF '{file_path}': {error_name}: {error_text}") from exc

    processed = []
    for doc in raw_docs:
        text = doc.page_content

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove non-printable / junk characters
        text = re.sub(r"[^\x00-\x7F]+", " ", text)

        # Skip very short pages (headers/footers only)
        if len(text) < 50:
            continue

        doc.page_content = text
        processed.append(doc)

    if not processed:
        raise ValueError(
            f"No usable text could be extracted from '{file_path}'. "
            "All pages were empty or below minimum content threshold."
        )

    print(f"Loaded {len(raw_docs)} pages -> {len(processed)} after preprocessing")
    return processed
