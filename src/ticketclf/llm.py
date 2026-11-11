"""Zero-shot LLM classifier for comparison against the TF-IDF model.

This module was NOT run to produce the results in the README (it needs an API key and
network access). Run it yourself to fill in the LLM row of the comparison:

    export ANTHROPIC_API_KEY=...
    PYTHONPATH=src python -m ticketclf.llm --sample 400

It scores the same "never-seen phrasings" test set the TF-IDF model struggles with,
caches every response to disk (re-runs are free), and writes reports/llm_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

from ticketclf.models import CATEGORIES

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5")
PROMPT = (Path(__file__).resolve().parents[2] / "prompts" / "classify_ticket.md").read_text()


def classify(text: str, model: str = DEFAULT_MODEL, retries: int = 3) -> dict:
    headers = {
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {"model": model, "max_tokens": 60, "temperature": 0,
            "messages": [{"role": "user", "content": PROMPT.replace("{ticket}", text)}]}
    for attempt in range(retries):
        r = requests.post(API_URL, headers=headers, json=body, timeout=60)
        if r.status_code == 200:
            out = r.json()
            raw = "".join(b.get("text", "") for b in out.get("content", []))
            return {**parse(raw), "input_tokens": out["usage"]["input_tokens"],
                    "output_tokens": out["usage"]["output_tokens"]}
        if r.status_code in (429, 500, 529):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError("LLM request failed after retries")


def parse(raw: str) -> dict:
    """Extract {"queue", "confidence"}; fall back safely if the model strays from the format."""
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    q = d.get("queue", "")
    if q not in CATEGORIES:
        q = next((c for c in CATEGORIES if c in raw), "unparseable")
    try:
        conf = float(d.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    return {"queue": q, "confidence": conf}


def main() -> None:
    from ticketclf.pipeline import split

    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/raw/tickets.csv"))
    p.add_argument("--sample", type=int, default=400)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--cache", type=Path, default=Path("data/llm_cache.jsonl"))
    a = p.parse_args()

    df = pd.read_csv(a.data)
    df["category_true"] = df["template_id"].str.split("#").str[0]
    _, _, unseen, _ = split(df)
    sample = unseen.sample(min(a.sample, len(unseen)), random_state=0)

    cache = {}
    if a.cache.exists():
        for line in a.cache.read_text().splitlines():
            rec = json.loads(line)
            cache[(rec["ticket_id"], rec["model"])] = rec
    rows = []
    with a.cache.open("a") as fh:
        for t in sample.itertuples():
            key = (t.ticket_id, a.model)
            if key not in cache:
                cache[key] = {"ticket_id": t.ticket_id, "model": a.model, **classify(t.text, a.model)}
                fh.write(json.dumps(cache[key]) + "\n")
            rows.append({**cache[key], "category": t.category})
    res = pd.DataFrame(rows)
    metrics = {
        "model": a.model,
        "n": len(res),
        "accuracy_unseen": float((res.queue == res.category).mean()),
        "unparseable_rate": float((res.queue == "unparseable").mean()),
        "avg_input_tokens": float(res.input_tokens.mean()),
        "avg_output_tokens": float(res.output_tokens.mean()),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/llm_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
