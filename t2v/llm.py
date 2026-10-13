"""Free-form request -> list of class names, using an OpenAI-compatible endpoint (AMD TokenFactory).

Configure with environment variables (never commit the key):
    TOKENFACTORY_API_KEY   your key
    TOKENFACTORY_BASE_URL  e.g. https://<tokenfactory-host>/v1  (copy it from the TokenFactory page)
    TOKENFACTORY_MODEL     a chat model id listed by the endpoint
Without them, the prompt is split on commas / "and" / "و" instead.
"""
import json
import os
import re

import requests

SYSTEM = (
    "You convert a request for an image classifier into the classes it must recognise. "
    "Reply with JSON only, no prose: {\"classes\": [\"...\"], \"task\": \"one short sentence\"}. "
    "Rules: between 2 and {k} classes; each class is a short singular English noun phrase (1-3 words) naming "
    "something visually recognisable in a photo; class names are ALWAYS written in English, even when the request is in Arabic or another language; "
    "no duplicates; keep the user's order when they list classes."
)


def _fallback(prompt, k):
    parts = re.split(r"\s*(?:,|،|;|\n|\band\b|\bor\b|\s+و\s+|\s+أو\s+)\s*", prompt)
    names = []
    for part in parts:
        part = part.strip(" .:-").lower()
        if part and part not in names:
            names.append(part)
    return names[:k]


def parse_classes(prompt, max_classes=10, timeout=60):
    key = os.environ.get("TOKENFACTORY_API_KEY")
    base = os.environ.get("TOKENFACTORY_BASE_URL")
    model = os.environ.get("TOKENFACTORY_MODEL")
    if not (key and base and model):
        return _fallback(prompt, max_classes), {"source": "fallback"}
    r = requests.post(
        base.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "temperature": 0,
              "messages": [{"role": "system", "content": SYSTEM.replace("{k}", str(max_classes))},
                           {"role": "user", "content": f"Request: {prompt}\n\nAnswer with English class names."}]},
        timeout=timeout,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", text, re.S)
    data = json.loads(match.group(0) if match else text)
    names = []
    for c in data.get("classes", []):
        c = str(c).strip().lower()
        if c and c not in names:
            names.append(c)
    return names[:max_classes], {"source": model, "task": data.get("task", "")}
