"""
Static catalog of PwnzzAI challenge surfaces for E2E solvability / contract tests.

Pairings (path, allowed HTTP statuses) cover unauthenticated browser entry.
"""

# GET routes that should stay reachable (HTML or JSON) for the lab shell.
GET_CHALLENGE_SURFACES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("/", (200,)),
    ("/login", (200,)),
    ("/basics", (200,)),
    ("/model-theft", (200,)),
    ("/supply-chain", (200,)),
    ("/insecure-plugin", (200,)),
    ("/sensitive-info", (200,)),
    ("/data-poisoning", (200,)),
    ("/data-poisoning/catering-rag", (200,)),
    ("/dos-attack", (200,)),
    ("/real-dos-attack", (200,)),
    ("/excessive-agency", (200,)),
    ("/agentic-tools", (200,)),
    ("/misinformation", (200,)),
    ("/glossary", (200,)),
    ("/direct-prompt-injection", (200,)),
    ("/indirect-prompt-injection", (200,)),
    ("/promotion-photo-claim", (200,)),
    ("/customer-support-safety", (200,)),
    ("/demo-malicious-model", (200,)),
    ("/pizza/1", (200, 404)),  # seed data may differ; 404 still proves routing
    ("/orders", (200, 302)),  # unauthenticated login flow can be 302 or final login page 200
    ("/check-openai-api-key", (200,)),
    ("/generate_sentiment_model", (200,)),
)

# Ollama direct prompt injection: level → substring that indicates a successful naive extract.
OLLAMA_DIRECT_LEVEL_SECRETS: dict[str, str] = {
    "1": "cheese",
    "2": "oven",
    "3": "olives",
    "4": "mushroom",
}

# Same coupon words for cloud direct-injection levels (OpenAI-compatible stack).
OPENAI_DIRECT_LEVEL_SECRETS: dict[str, str] = {**OLLAMA_DIRECT_LEVEL_SECRETS, "5": "mozzarella"}

# RAG / misinformation refresh endpoints (POST, empty JSON body).
RAG_REFRESH_POST_PATHS: tuple[str, ...] = (
    "/update-rag-ollama",
    "/update-rag-misinformation",
    "/update-rag-openai",
    "/update-rag-openai-misinfo",
)
