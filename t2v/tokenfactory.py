"""AMD TokenFactory: the LLM endpoint behind every agent step of T2V (OpenAI-compatible chat completions).

All agent roles go through `chat` here:
    1 understand  t2v.llm.parse_classes   free-form request (any language) -> English class names
    2 ground      t2v.ground.ground        user's words -> the generator's vocabulary (same category only)
    3 describe    t2v.describe.describe    class name -> visual descriptions (CuPL)
    4 analogise   t2v.neighbors.ask        never-seen class -> weighted look-alike known classes

Only the key is required (never commit it):
    TOKENFACTORY_API_KEY    your key
    TOKENFACTORY_BASE_URL   optional, defaults to DEFAULT_BASE_URL
    TOKENFACTORY_MODEL      optional, defaults to DEFAULT_MODEL
"""
import os
import time

import requests

DEFAULT_BASE_URL = "https://developer.amd.com.cn/radeon/api/v1"
DEFAULT_MODEL = "DeepSeek-V4-Flash"

STATS = {"calls": 0, "seconds": 0.0}  # TokenFactory calls made by this process


def config():
    """(key, base_url, model), or None when no key is set (the agent then runs offline)."""
    key = os.environ.get("TOKENFACTORY_API_KEY")
    if not key:
        return None
    return key, os.environ.get("TOKENFACTORY_BASE_URL") or DEFAULT_BASE_URL, os.environ.get("TOKENFACTORY_MODEL") or DEFAULT_MODEL


def model_name():
    cfg = config()
    return cfg[2] if cfg else None


def chat(messages, temperature=0.0, max_tokens=3000, timeout=120):
    """One JSON-mode chat completion on TokenFactory; returns the reply text, or None without a key."""
    cfg = config()
    if cfg is None:
        return None
    key, base, model = cfg
    t0 = time.time()
    r = requests.post(base.rstrip("/") + "/chat/completions", headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "temperature": temperature, "max_tokens": max_tokens,
                            "response_format": {"type": "json_object"}, "messages": messages}, timeout=timeout)
    r.raise_for_status()
    STATS["calls"] += 1
    STATS["seconds"] += time.time() - t0
    return r.json()["choices"][0]["message"]["content"]
