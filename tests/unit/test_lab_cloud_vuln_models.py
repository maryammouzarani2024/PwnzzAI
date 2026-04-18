"""Ensure vulnerability modules pass LAB_CLOUD_* model env into chat_completion."""

import importlib


def _reload_provider_and_dos(monkeypatch):
    monkeypatch.setenv("LAB_CLOUD_LLM_MODEL", "anthropic/claude-3-5-sonnet-20240620")
    import application.provider_config as pc
    import application.vulnerabilities.openai_dos as dos

    importlib.reload(pc)
    return importlib.reload(dos)


def test_openai_dos_uses_lab_cloud_llm_model_default(monkeypatch):
    dos = _reload_provider_and_dos(monkeypatch)
    captured = {}

    def fake_chat_completion(*args, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(dos, "chat_completion", fake_chat_completion)
    dos.chat_with_openai("hello", "sk-fake")
    assert captured["model"] == "anthropic/claude-3-5-sonnet-20240620"


def test_openai_excessive_agency_uses_excessive_agency_env(monkeypatch):
    monkeypatch.setenv("LAB_CLOUD_LLM_MODEL_EXCESSIVE_AGENCY", "gemini/gemini-2.0-flash")
    import application.provider_config as pc
    import application.vulnerabilities.openai_excessive_agency as ea

    importlib.reload(pc)
    ea = importlib.reload(ea)
    captured = {}

    def fake_chat_completion(*args, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(ea, "chat_completion", fake_chat_completion)
    ea.openai_chat("order pizza", "sk-fake")
    assert captured["model"] == "gemini/gemini-2.0-flash"
