def _reload_llm_chat(monkeypatch):
    monkeypatch.delenv("LITELLM_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    import application.llm_chat as llm_chat

    return llm_chat


def test_normalize_litellm_model_defaults_to_resolved(monkeypatch):
    llm_chat = _reload_llm_chat(monkeypatch)
    assert llm_chat.normalize_litellm_model(None) == "openai/gpt-4o-mini"
    assert llm_chat.normalize_litellm_model("") == "openai/gpt-4o-mini"


def test_normalize_litellm_model_bare_name_gets_openai_prefix(monkeypatch):
    llm_chat = _reload_llm_chat(monkeypatch)
    assert llm_chat.normalize_litellm_model("gpt-3.5-turbo") == "openai/gpt-3.5-turbo"


def test_normalize_litellm_model_passes_through_prefixed(monkeypatch):
    monkeypatch.setenv("LITELLM_MODEL", "gemini/gemini-2.0-flash")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    import application.llm_chat as llm_chat
    assert llm_chat.normalize_litellm_model("gemini/gemini-1.5-flash") == "gemini/gemini-1.5-flash"


def test_chat_completion_passes_timeout_and_api_key(monkeypatch):
    llm_chat = _reload_llm_chat(monkeypatch)
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return type(
            "R",
            (),
            {
                "choices": [
                    type(
                        "C",
                        (),
                        {"message": type("M", (), {"content": "hello"})()},
                    )()
                ]
            },
        )()

    monkeypatch.setattr(llm_chat.litellm, "completion", fake_completion)
    out = llm_chat.chat_completion(
        [{"role": "user", "content": "x"}],
        api_key="sk-test",
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=10,
    )
    assert out == "hello"
    assert captured["model"] == "openai/gpt-3.5-turbo"
    assert captured["api_key"] == "sk-test"
    assert captured["temperature"] == 0.5
    assert captured["max_tokens"] == 10
    assert captured["messages"] == [{"role": "user", "content": "x"}]
    assert "timeout" in captured
