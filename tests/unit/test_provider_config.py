import importlib


def _reload_provider_config():
    import application.provider_config as provider_config
    return importlib.reload(provider_config)


def test_get_openai_api_key_prefers_session_when_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("ALLOW_PRECONFIGURED_OPENAI_KEY", "true")
    monkeypatch.setenv("PREFER_SESSION_OPENAI_KEY", "true")
    provider_config = _reload_provider_config()
    assert provider_config.get_openai_api_key({"openai_api_key": "sk-session"}) == "sk-session"


def test_get_openai_api_key_uses_env_when_session_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("ALLOW_PRECONFIGURED_OPENAI_KEY", "true")
    monkeypatch.setenv("PREFER_SESSION_OPENAI_KEY", "true")
    provider_config = _reload_provider_config()
    assert provider_config.get_openai_api_key({}) == "sk-env"


def test_resolve_provider_auto_with_key(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "auto")
    monkeypatch.setenv("ENABLE_PROVIDER_FALLBACK", "true")
    provider_config = _reload_provider_config()
    assert provider_config.resolve_provider(has_openai_key=True) == "openai"


def test_resolve_provider_auto_without_key(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "auto")
    monkeypatch.setenv("ENABLE_PROVIDER_FALLBACK", "true")
    provider_config = _reload_provider_config()
    assert provider_config.resolve_provider(has_openai_key=False) == "ollama"


def test_provider_snapshot_has_expected_fields(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "mistral:7b")
    provider_config = _reload_provider_config()
    snapshot = provider_config.provider_snapshot()
    assert snapshot["model_provider"] == "ollama"
    assert snapshot["ollama_model"] == "mistral:7b"
