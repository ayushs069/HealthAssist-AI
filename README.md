# HealthAssist AI 🏥

A production-ready **Healthcare RAG (Retrieval-Augmented Generation)** system powered by a FastAPI backend, a modern HTML/CSS/JS dashboard, and LLMs via Groq — built to answer medical questions from a curated knowledge base.

---

## 🚀 Features

- **RAG Pipeline** — PDF ingestion → Chunking → Embeddings → FAISS Vector Search → LLM Answer
- **Live Pipeline Visualization** — See each step animate as queries are processed
- **Multi-LLM Support** — Groq (LLaMA 3, LLaMA 3.1), Ollama (local), or HuggingFace
- **Multi-Vector Store** — FAISS (default) or ChromaDB
- **Configurable Chunking & Embedding** — Swap models/strategies from the UI
- **Chat History** — Auto-saved locally with export support
- **Evaluation Panel** — Compare RAG configurations by ROUGE, BLEU, and response time
- **Delete Messages** — Hover-to-reveal trash icon on each chat bubble
- **Professional Dark Dashboard** — Royal Blue & Amethyst theme

---

## 📁 Project Structure

```
HealthRag/
├── api.py                  # FastAPI backend (main entry point)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
│
├── src/                    # Core RAG pipeline modules
│   ├── loader.py           # PDF document loading
│   ├── chunking.py         # Text splitting strategies
│   ├── embeddings.py       # Embedding model wrapper
│   ├── vectorstores.py     # FAISS / ChromaDB abstraction
│   ├── llm.py              # LLM provider abstraction (Groq, Ollama, HF)
│   ├── prompts.py          # Prompt templates
│   ├── rag_pipeline.py     # Full pipeline orchestration
│   ├── evaluation.py       # ROUGE, BLEU, BERTScore evaluation
│   └── memory_utils.py     # Memory tracking utilities
│
├── frontend/               # Web dashboard (vanilla HTML/CSS/JS)
│   ├── index.html
│   ├── styles.css
│   └── script.js
│
├── data/                   # Knowledge base documents
│   └── qa_pairs/           # QA pairs for evaluation
│
└── experiments/            # Retrieval & chunking experiments
    ├── experiment_1.py
    └── experiment_2.py
```

---

## ⚙️ Setup

### 1. Clone & Create Virtual Environment

```bash
git clone https://github.com/<your-username>/HealthRag.git
cd HealthRag
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```env
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
```

> Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Add Your Knowledge Base

Place your PDF documents in `data/documents/` (or use the default `healthcare.pdf`).

### 5. Run the Application

```bash
python api.py
```

Open your browser at **http://localhost:8000**

---

## 🧪 Running Experiments

```bash
python experiments/experiment_1.py
python experiments/experiment_2.py
```

---

## 📊 Evaluation

```bash
python evaluate.py
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| RAG Framework | LangChain |
| Vector Store | FAISS / ChromaDB |
| Embeddings | sentence-transformers (MiniLM, BGE) |
| LLMs | Groq (LLaMA 3), Ollama (local), HuggingFace |
| Frontend | HTML5, CSS3, Vanilla JS |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
