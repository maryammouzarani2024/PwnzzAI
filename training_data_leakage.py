"""
Training-data leakage demo helpers.

The HuggingFace tab calls ``huggingface_leak_endpoint`` from this module. The full
HF inference stack is not bundled in the default container image; this stub keeps
the route importable and returns a clear JSON contract for labs and E2E probes.
"""

from __future__ import annotations

from flask import jsonify, request


def huggingface_leak_endpoint():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return (
            jsonify(
                {
                    "error": "No query provided",
                    "response": "",
                    "has_leakage": False,
                    "leaked_info": [],
                }
            ),
            400,
        )
    return jsonify(
        {
            "response": (
                "Hugging Face on-device inference is not configured in this image. "
                "Use the Ollama or cloud provider tabs for the live training-data "
                "leakage lab."
            ),
            "has_leakage": False,
            "leaked_info": [],
            "model_type": "stub",
        }
    )
