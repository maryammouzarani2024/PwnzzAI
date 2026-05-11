"""Render direct-prompt escalation ladder (B0–B9) system prompts from Jinja2 templates."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _template_dir() -> Path:
    return _repo_root() / "prompts" / "direct_prompt_escalation"


@lru_cache(maxsize=1)
def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_template_dir())),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_b_stream_system_prompt(stage: int, *, secret: str, baseline_level: str) -> str:
    """
    Load ``b{stage}.jinja2`` (e.g. ``b00.jinja2`` for B0) and render with the given coupon secret.

    ``baseline_level`` is the legacy 1–5 difficulty key used for catalog/tests; templates may ignore it.
    """
    s = max(0, min(9, stage))
    name = f"b{s:02d}.jinja2"
    tmpl = _environment().get_template(name)
    return tmpl.render(secret=secret, stage=s, baseline_level=baseline_level).strip()
