"""Free-form request -> list of class names, using an OpenAI-compatible endpoint (AMD TokenFactory).

Configure with environment variables (never commit the key):
    TOKENFACTORY_API_KEY   your key
    TOKENFACTORY_BASE_URL  e.g. https://<tokenfactory-host>/v1  (copy it from the TokenFactory page)
    TOKENFACTORY_MODEL     a chat model id listed by the endpoint
Without them, the prompt is split on commas / "and" / "و" instead, and request phrasing ("I want a model that\nseparates ...") and plurals are stripped.
"""
import json
import os
import re

import requests

SYSTEM = (
    "You convert a request for an image classifier into the classes it must recognise. "
    "Reply with JSON only, no prose: {\"classes\": [\"...\"], \"task\": \"one short sentence\", "
    "\"descriptions\": {\"<class>\": [\"...\", \"...\", \"...\"]}}. "
    "Rules: between 2 and {k} classes; each class is a short singular English noun phrase (1-3 words) naming "
    "something visually recognisable in a photo; class names are ALWAYS written in English, even when the request is in Arabic or another language; "
    "no duplicates; keep the user's order when they list classes; for each class give 3 short visual "
    "descriptions (8-15 words) of how it looks in a photo, stressing what tells it apart from the other "
    "classes in this request."
)


# words that only phrase the request ("I want a model that separates ... from ..."), dropped offline
_FILLER = set("""i we you me us please can could would will want wanna need needs like id i'd make build create give
get train a an the some my our model models classifier classifiers network ai app system that which who to for
separate separates distinguish distinguishes differentiate tell tells apart classify classifies recognise recognize
recognises recognizes detect detects identify identifies sort sorts between from vs versus among images image photos
photo pictures picture of kinds types""".split())


def _singular(word):
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and word[:-2].endswith(("s", "x", "ch", "sh")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _fallback(prompt, k):
    parts = re.split(r"\s*(?:,|،|;|\n|\band\b|\bor\b|\bvs\.?\b|\bversus\b|\bfrom\b|\s+و\s+|\s+أو\s+)\s*", prompt)
    names = []
    for part in parts:
        words = re.sub(r"[^\w\s'-]", " ", part.lower()).split()
        while words and words[0] in _FILLER:
            words.pop(0)
        while words and words[-1] in _FILLER:
            words.pop()
        if not words:
            continue
        words[-1] = _singular(words[-1])
        name = " ".join(words)
        if name not in names:
            names.append(name)
    return names[:k]


def parse_classes(prompt, max_classes=10, timeout=60):
    key = os.environ.get("TOKENFACTORY_API_KEY")
    base = os.environ.get("TOKENFACTORY_BASE_URL")
    model = os.environ.get("TOKENFACTORY_MODEL")
    if not (key and base and model):
        return _fallback(prompt, max_classes), {"source": "fallback"}
    data = None
    for attempt in range(2):  # JSON mode makes broken output rare; retry once if it still happens
        r = requests.post(
            base.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "temperature": 0, "max_tokens": 2000, "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": SYSTEM.replace("{k}", str(max_classes))},
                               {"role": "user", "content": f"Request: {prompt}\n\nAnswer with English class names."}]},
            timeout=timeout,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", text, re.S)
        try:
            data = json.loads(match.group(0) if match else text)
            break
        except json.JSONDecodeError:
            continue
    if data is None:
        return _fallback(prompt, max_classes), {"source": "fallback (unparseable LLM reply)"}
    names = []
    for c in data.get("classes", []):
        c = str(c).strip().lower()
        if c and c not in names:
            names.append(c)
    raw = {str(k).strip().lower(): v for k, v in (data.get("descriptions") or {}).items()}
    descriptions = {n: [str(d) for d in raw[n]] for n in names[:max_classes] if isinstance(raw.get(n), list)}
    return names[:max_classes], {"source": model, "task": data.get("task", ""), "descriptions": descriptions}
