#!/usr/bin/env python3
"""
Generate a J.League community overview JSON using Gemini API.

Usage:
  GEMINI_API_KEY=... python3 scripts/generate_jleague_overview.py

Output:
  output/jleague_overview.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, List

import requests
from zoneinfo import ZoneInfo

OUTPUT_PATH = Path("output/jleague_overview.json")
DEFAULT_MODEL = "gemini-2.5-flash"
MODEL_NAME = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
API_VERSION = "v1beta"
API_BASE = "https://generativelanguage.googleapis.com"
LOG_PATH = Path("output/jleague_overview_errors.log")
TZ = ZoneInfo("Asia/Tokyo")

SIMILARITY_THRESHOLD = 0.72
MAX_RETRIES = 4


def _build_prompt(now: datetime, recent_summaries: List[str]) -> str:
    date_str = now.strftime("%Y年%m月%d日")
    lines = [
        f"あなたが知っている最新のJリーグに関するニュースや話題を1つ選び（{date_str}以前の情報で構わない）、",
        "1〜2文の自然な日本語にまとめてください。",
        "「今日のJリーグのニュースでは」などの前置きは不要。",
        "文末は「。」で終わること。",
    ]
    if recent_summaries:
        lines.append("\n以下の話題はすでに取り上げ済みなので別の話題を選んでください:")
        lines.extend(f"- {s}" for s in recent_summaries)
    return "\n".join(lines)


def _now_iso() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def _load_existing() -> dict[str, Any]:
    if not OUTPUT_PATH.exists():
        return {}
    try:
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("warning: existing JSON is invalid; starting fresh", file=sys.stderr)
        return {}
    except OSError as exc:
        print(f"warning: failed to read existing JSON: {exc}; starting fresh", file=sys.stderr)
        return {}


def _is_complete_sentence(text: str) -> bool:
    s = text.strip()
    return bool(s) and s.endswith(("。", "！", "？", "!", "?"))


def _build_endpoint(model_name: str, api_version: str) -> str:
    return f"{API_BASE}/{api_version}/models/{model_name}:generateContent"


def _log_error(message: str, detail: str) -> None:
    ts = datetime.now(TZ).isoformat(timespec="seconds")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")
        if detail:
            f.write(f"{detail}\n")


def _normalize_text(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _is_similar_topic(summary: str, candidates: List[str]) -> bool:
    base = _normalize_text(summary)
    if not base:
        return False
    for prev in candidates:
        prev_n = _normalize_text(prev)
        if not prev_n:
            continue
        ratio = SequenceMatcher(None, base, prev_n).ratio()
        if ratio >= SIMILARITY_THRESHOLD:
            return True
    return False


def _call_gemini(api_key: str, recent_summaries: List[str], now: datetime) -> str:
    def build_payload(prompt_text: str) -> dict[str, Any]:
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 512,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

    def post_with_timeout(endpoint: str, payload: dict[str, Any]) -> requests.Response:
        try:
            return requests.post(endpoint, params={"key": api_key}, json=payload, timeout=60)
        except requests.exceptions.ReadTimeout:
            return requests.post(endpoint, params={"key": api_key}, json=payload, timeout=120)

    endpoint = _build_endpoint(MODEL_NAME, API_VERSION)
    recent = list(recent_summaries)

    for attempt in range(MAX_RETRIES):
        prompt_text = _build_prompt(now, recent[-20:])
        payload = build_payload(prompt_text)
        resp = post_with_timeout(endpoint, payload)

        if os.environ.get("GEMINI_DEBUG") == "1":
            print(f"debug: model={MODEL_NAME} attempt={attempt + 1}", file=sys.stderr)

        if resp.status_code >= 400:
            print(f"error: Gemini API status {resp.status_code}", file=sys.stderr)
            print(resp.text, file=sys.stderr)
            _log_error(f"Gemini API status {resp.status_code}", resp.text)
            resp.raise_for_status()

        data = resp.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            # Skip thought parts (gemini-2.5-flash thinking model)
            text = next(
                (p["text"] for p in parts if isinstance(p, dict) and not p.get("thought") and p.get("text", "").strip()),
                None,
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected Gemini response format: {exc}")

        if os.environ.get("GEMINI_DEBUG") == "1":
            print("debug: raw text:", repr(text), file=sys.stderr)

        if not text or not text.strip():
            _log_error(f"warning: empty response on attempt {attempt + 1}", "")
            time.sleep(min(2 ** attempt, 8))
            continue

        summary = text.strip()

        if not _is_complete_sentence(summary):
            _log_error(f"warning: incomplete sentence on attempt {attempt + 1}", summary[:500])
            recent.append(summary)
            continue

        if _is_similar_topic(summary, recent_summaries):
            _log_error(f"warning: duplicate topic on attempt {attempt + 1}", summary[:500])
            recent.append(summary)
            continue

        return summary

    raise ValueError(f"failed to generate valid summary after {MAX_RETRIES} attempts")


def _prune_history(history: List[dict[str, Any]], cutoff: datetime) -> List[dict[str, Any]]:
    kept: List[dict[str, Any]] = []
    seen: set[str] = set()
    for item in history:
        ts = item.get("timestamp")
        if not ts:
            continue
        summary_text = item.get("summary_text")
        if not isinstance(summary_text, str) or not summary_text.strip():
            continue
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            continue
        if dt >= cutoff:
            normalized = " ".join(summary_text.split())
            if normalized in seen:
                continue
            seen.add(normalized)
            kept.append(item)
    return kept


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("error: GEMINI_API_KEY is not set", file=sys.stderr)
        return 2

    existing = _load_existing()
    history = existing.get("history", []) if isinstance(existing, dict) else []
    if not isinstance(history, list):
        history = []

    prev_current = existing.get("current") if isinstance(existing, dict) else None
    if isinstance(prev_current, dict):
        if "timestamp" in prev_current and isinstance(prev_current.get("summary_text"), str) and prev_current.get("summary_text").strip():
            history.append(prev_current)

    now = datetime.now(TZ)
    now_iso = now.isoformat(timespec="seconds")
    cutoff = now - timedelta(hours=72)
    history = _prune_history(history, cutoff)
    recent_summaries = [h.get("summary_text", "") for h in history[-20:] if isinstance(h.get("summary_text"), str)]
    if isinstance(prev_current, dict) and isinstance(prev_current.get("summary_text"), str):
        recent_summaries.append(prev_current["summary_text"])

    summary = _call_gemini(api_key, recent_summaries, now)

    output = {
        "updated_at": now_iso,
        "current": {
            "timestamp": now_iso,
            "summary_text": summary,
        },
        "history": history,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
