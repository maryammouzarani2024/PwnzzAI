import os
from typing import Optional


def _get_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "auto").strip().lower()
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", os.environ.get("OLLAMA_FALLBACK_MODEL", "llama3.2:1b")).strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
MODEL_TIMEOUT_SECONDS = int(os.environ.get("MODEL_TIMEOUT_SECONDS", "60"))
ENABLE_PROVIDER_FALLBACK = _get_bool("ENABLE_PROVIDER_FALLBACK", True)
ALLOW_PRECONFIGURED_OPENAI_KEY = _get_bool("ALLOW_PRECONFIGURED_OPENAI_KEY", True)
PREFER_SESSION_OPENAI_KEY = _get_bool("PREFER_SESSION_OPENAI_KEY", True)


def get_openai_api_key(session_obj) -> str:
    session_key = (session_obj.get("openai_api_key", "") if session_obj else "").strip()
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if PREFER_SESSION_OPENAI_KEY and session_key:
        return session_key
    if ALLOW_PRECONFIGURED_OPENAI_KEY and env_key:
        return env_key
    if session_key:
        return session_key
    return ""


def resolve_provider(preferred: Optional[str] = None, has_openai_key: bool = False) -> str:
    requested = (preferred or MODEL_PROVIDER).strip().lower()

    if requested in {"ollama", "openai"}:
        return requested

    if requested != "auto":
        requested = "auto"

    if has_openai_key:
        return "openai"
    if ENABLE_PROVIDER_FALLBACK:
        return "ollama"
    return "openai"


def provider_snapshot() -> dict:
    return {
        "model_provider": MODEL_PROVIDER,
        "ollama_host": OLLAMA_HOST,
        "ollama_model": OLLAMA_MODEL,
        "openai_model": OPENAI_MODEL,
        "model_timeout_seconds": MODEL_TIMEOUT_SECONDS,
        "allow_preconfigured_openai_key": ALLOW_PRECONFIGURED_OPENAI_KEY,
        "prefer_session_openai_key": PREFER_SESSION_OPENAI_KEY,
        "enable_provider_fallback": ENABLE_PROVIDER_FALLBACK,
    }
