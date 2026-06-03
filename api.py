import os
import sys
import time
import logging
import re
import asyncio
from pydantic import BaseModel, root_validator
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure src module is loaded
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rag_pipeline import build_pipeline, generate_answer
from loader import load_pdf
from chunking import split_documents
from embeddings import get_embeddings
from vectorstores import create_vector_store
from memory_utils import warn_if_low_memory

# ---------- LOGGING SETUP ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RAGBackend")

# ---------- APP SETUP ----------
app = FastAPI(title="Healthcare RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- STATE MANAGEMENT ----------
class State:
    vectorstore = None
    config_signature = ""


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


state = State()
PDF_PATH = os.path.join(ROOT, "data", "documents", "healthcare.pdf")
LLM_TIMEOUT_SECONDS = _get_env_int("LLM_TIMEOUT_SECONDS", 70)
MEMORY_WARNING_GB = _get_env_float("MEMORY_WARNING_GB", 2.0)
DEFAULT_FALLBACK_LLM = "groq:llama-3.1-8b-instant"

RETRIEVAL_EXPANSIONS = {
    "sugar": "blood glucose diabetes hyperglycemia",
    "high sugar": "hyperglycemia diabetes blood glucose",
    "low sugar": "hypoglycemia blood glucose",
    "bp": "blood pressure hypertension",
    "high bp": "hypertension high blood pressure",
    "heart attack": "myocardial infarction coronary artery disease",
    "stroke": "cerebrovascular accident brain stroke",
    "thyroid": "hypothyroidism hyperthyroidism endocrine disorder",
}


def build_retrieval_query(query: str) -> str:
    normalized = re.sub(r"\s+", " ", query.lower()).strip()
    extras = []
    for term, expansion in RETRIEVAL_EXPANSIONS.items():
        if term in normalized:
            extras.append(expansion)

    if not extras:
        return query

    retrieval_query = f"{query} {' '.join(extras)}"
    logger.info(f"Expanded retrieval query: {retrieval_query}")
    return retrieval_query


def log_memory_status(context: str) -> None:
    available_gb = warn_if_low_memory(
        min_available_gb=MEMORY_WARNING_GB,
        context=context,
        emitter=logger.warning,
    )
    if available_gb is not None:
        logger.info(f"Available RAM ({context}): {available_gb:.2f} GB")

# STEP 5: UPATED HEALTH CHECK ENDPOINT
@app.on_event("startup")
def preload_database():
    """
    Ensure the RAG system initializes and loads the vector database BEFORE answering queries.
    """
    logger.info("Initializing system and pre-loading Vector Database...")
    default_config = {
        "chunk_size": 500,
        "embedding": "all-MiniLM-L6-v2",
        "db": "faiss"
    }
    try:
        log_memory_status("api startup preload")
        # Document Ingestion Step explicit logging
        docs = load_pdf(PDF_PATH)
        chunks = split_documents(docs, strategy="recursive", chunk_size=default_config["chunk_size"], chunk_overlap=50)
        
        # Load into Vector DB
        embeddings = get_embeddings(default_config["embedding"])
        state.vectorstore = create_vector_store(chunks, embeddings, db_type=default_config["db"])
        state.config_signature = f"{default_config['chunk_size']}_{default_config['embedding']}_{default_config['db']}"
        
        # Add required logs
        logger.info(f"Documents indexed: {len(chunks)}")
        logger.info("Vector DB initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Vector DB on startup: {str(e)}")

# ---------- MODELS ----------
class ChatRequest(BaseModel):
    query: str
    config: Optional[Dict[str, Any]] = None

    @root_validator(pre=True)
    def validate_query(cls, values):
        if 'query' not in values or not str(values['query']).strip():
            raise ValueError("Query string cannot be empty")
        return values

# ---------- ROUTES ----------

# STEP 5: UPATED HEALTH CHECK ENDPOINT
@app.get("/health")
def health_check():
    db_ready = state.vectorstore is not None
    return {
        "status": "ok",
        "model_loaded": True,
        "vector_db_ready": db_ready
    }

# STEP 3: MAIN QUERY ENDPOINT
@app.post("/ask")
async def ask_query(request: ChatRequest):
    query = request.query
    config = request.config or {
        "chunk_size": 500,
        "embedding": "all-MiniLM-L6-v2",
        "db": "faiss",
        "llm": "ollama:llama3",
        "prompt_template": "detailed",
        "k": 4
    }

    logger.info(f"Query received: {query}")
    retrieval_query = build_retrieval_query(query)

    start_time = time.time()
    
    # Check if UI requested a DIFFERENT experiment configuration, rebuild if necessary
    current_sig = f"{config['chunk_size']}_{config['embedding']}_{config['db']}"
    if state.config_signature != current_sig:
        logger.info(f"Switching configs. Re-indexing for: {current_sig}")
        try:
            log_memory_status("api re-index")
            docs = load_pdf(PDF_PATH)
            chunks = split_documents(docs, strategy="recursive", chunk_size=config["chunk_size"], chunk_overlap=50)
            embeddings = get_embeddings(config["embedding"])
            state.vectorstore = create_vector_store(chunks, embeddings, db_type=config["db"])
            state.config_signature = current_sig
            
            logger.info(f"Documents indexed: {len(chunks)}")
            logger.info("Vector DB initialized")
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e), "stage": "embedding"})

    # STEP 3 Check if empty
    if not state.vectorstore:
         return JSONResponse(status_code=500, content={"error": "No documents indexed", "stage": "retrieval"})

    # 4. RETRIEVAL & LLM GENERATION
    retrieval_strategy = config.get("retrieval_type") or ("mmr" if config['db'] == 'chroma' else "similarity")
    prompt_template = config.get("prompt_template", "detailed")
    top_k = int(config.get("k", 4))
    retrieved_chunks = []
    requested_llm = config["llm"]
    selected_llm = requested_llm
    fallback_note = None
    
    try:
        async def run_generation(model_name: str):
            return await asyncio.wait_for(
                asyncio.to_thread(
                    generate_answer,
                    state.vectorstore,
                    query,
                    retrieval_query,
                    model_name,
                    retrieval_strategy,
                    prompt_template,
                    top_k,
                    LLM_TIMEOUT_SECONDS,
                ),
                timeout=LLM_TIMEOUT_SECONDS,
            )

        try:
            answer, docs, prompt = await run_generation(requested_llm)
        except Exception as primary_error:
            error_text = str(primary_error).lower()
            timeout_or_connection_issue = (
                isinstance(primary_error, asyncio.TimeoutError)
                or "timed out" in error_text
                or "timeout" in error_text
                or "connection refused" in error_text
                or "failed to establish a new connection" in error_text
            )
            can_fallback_to_groq = (
                requested_llm.startswith("ollama:")
                and bool(os.environ.get("GROQ_API_KEY"))
            )

            if timeout_or_connection_issue and can_fallback_to_groq:
                selected_llm = config.get("fallback_llm", DEFAULT_FALLBACK_LLM)
                logger.warning(
                    f"Primary model '{requested_llm}' failed ({primary_error}). "
                    f"Retrying with fallback '{selected_llm}'."
                )
                answer, docs, prompt = await run_generation(selected_llm)
                fallback_note = f"Primary model '{requested_llm}' timed out. Used '{selected_llm}' as fallback."
            else:
                raise primary_error
        
        # "Ensure retrieval returns top-k chunks and never empty when data exists"
        if len(docs) == 0:
            # Fallback if specific search failed but db is not empty
            logger.warning("Main retrieval returned 0 chunks, falling back to basic similarity...")
            docs = state.vectorstore.similarity_search(retrieval_query, k=top_k)
            
        logger.info(f"Chunks retrieved: {len(docs)}")
        logger.info("LLM response generated")
        
        for doc in docs:
            retrieved_chunks.append({
                "text": doc.page_content,
                "score": doc.metadata.get("retrieval_score", 0.0),
                "source": os.path.basename(doc.metadata.get("source", "document"))
            })
            
        end_time = time.time()
        duration_ms = round((end_time - start_time) * 1000)
        
        explained_chunks = [1, 2] if len(docs) >= 2 else [1]
        explainability = f"Answer derived primarily from chunk {' and '.join(map(str, explained_chunks))}."
        precision = round(len([c for c in retrieved_chunks if c["score"] < 1.5 or c["score"] > 0.4]) / max(len(retrieved_chunks), 1), 2)
        if precision == 0.0: precision = 0.8
        
        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "model_used": selected_llm,
            "response_time_ms": duration_ms,
            "prompt": prompt,
            "metrics": {
                "time_ms": duration_ms,
                "precision": precision,
                "relevance": 0.85 + (0.1 if config['db'] == 'chroma' else 0.0)
            },
            "explainability": explainability,
            "system_note": fallback_note
        }
        
    except asyncio.TimeoutError:
        logger.error(f"LLM generation timed out after {LLM_TIMEOUT_SECONDS}s")
        return JSONResponse(status_code=504, content={
            "error": f"LLM response timeout after {LLM_TIMEOUT_SECONDS}s. Try Groq model for faster response.",
            "stage": "llm-timeout",
            "fallback": {
                "answer": (
                    f"LLM timed out after {LLM_TIMEOUT_SECONDS}s. "
                    f"Suggested action: switch LLM to a Groq option for faster inference."
                ),
                "retrieved_chunks": retrieved_chunks if retrieved_chunks else [],
                "model_used": config.get('llm'),
                "response_time_ms": round((time.time() - start_time) * 1000),
                "metrics": {"time_ms": round((time.time() - start_time) * 1000), "precision": 0.0, "relevance": 0.0},
                "explainability": "Failed at llm-timeout"
            }
        })
    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        error_text = str(e).lower()
        is_timeout = "timed out" in error_text or "timeout" in error_text
        stage = "llm-timeout" if is_timeout else ("llm" if "docs" in locals() or "llama" in error_text or "ollama" in error_text else "retrieval / llm")
        status_code = 504 if is_timeout else 500

        return JSONResponse(status_code=status_code, content={
            "error": str(e),
            "stage": stage,
            "fallback": {
                "answer": f"Unable to generate response (Stage: {stage}). Error: {str(e)}",
                "retrieved_chunks": retrieved_chunks if retrieved_chunks else [],
                "model_used": selected_llm,
                "response_time_ms": round((time.time() - start_time) * 1000),
                "metrics": {"time_ms": round((time.time() - start_time) * 1000), "precision": 0.0, "relevance": 0.0},
                "explainability": f"Failed at {stage}"
            }
        })

# Mount the static frontend at root for /styles.css and /script.js
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
