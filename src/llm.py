import os
from typing import Optional
from langchain_ollama import OllamaLLM

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # dotenv is optional at runtime; direct environment variables still work.
    pass


OLLAMA_MODELS = {
    "llama3": "Meta LLaMA 3 8B general-purpose model",
    "llama3.1": "Meta LLaMA 3.1 8B improved instruction following",
    "gemma2": "Google Gemma 2 9B efficient open model",
    "gemma": "Google Gemma 7B lightweight model",
    "mistral": "Mistral 7B high-quality open model",
}


def get_llm(
    model_spec: str = "ollama:llama3",
    timeout_seconds: int = 80,
    ollama_base_url: Optional[str] = None,
):
    """
    Return a LangChain-compatible LLM.
    Format: provider:model_name
    """
    if ":" in model_spec:
        provider, model_name = model_spec.split(":", 1)
    else:
        provider = "ollama"
        model_name = model_spec

    if provider == "ollama":
        desc = OLLAMA_MODELS.get(model_name, "custom Ollama model")
        print(f"LLM selected: {model_name} ({desc})")
        timeout_seconds = max(10, int(timeout_seconds))
        resolved_base_url = (
            ollama_base_url
            or os.environ.get("OLLAMA_BASE_URL", "").strip()
            or None
        )
        return OllamaLLM(
            model=model_name,
            base_url=resolved_base_url,
            sync_client_kwargs={"timeout": timeout_seconds},
            async_client_kwargs={"timeout": timeout_seconds},
        )

    if provider == "hf":
        try:
            from langchain_huggingface import HuggingFaceEndpoint

            token = os.environ.get("HF_TOKEN", "")
            if not token:
                print("HF_TOKEN is not set. Set it with: set HF_TOKEN=your_token")

            print(f"LLM selected: {model_name} (HuggingFace Inference API)")
            return HuggingFaceEndpoint(
                repo_id=model_name,
                huggingfacehub_api_token=token,
                max_new_tokens=512,
                temperature=0.1,
                task="text-generation",
            )
        except ImportError as exc:
            raise ImportError("Install langchain-huggingface: pip install langchain-huggingface") from exc

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq

            token = os.environ.get("GROQ_API_KEY", "")
            if not token:
                print("GROQ_API_KEY is not set. Set it in .env or system environment.")

            print(f"LLM selected: {model_name} (Groq)")
            return ChatGroq(
                model=model_name,
                groq_api_key=token,
                temperature=0.1,
            )
        except ImportError as exc:
            raise ImportError("Install langchain-groq: pip install langchain-groq") from exc

    raise ValueError(
        f"Unknown provider: '{provider}'. Use 'ollama', 'hf', or 'groq'.\n"
        f"Examples: 'ollama:llama3', 'hf:mistralai/Mistral-7B-Instruct-v0.3', "
        f"'groq:llama-3.3-70b-versatile'"
    )
