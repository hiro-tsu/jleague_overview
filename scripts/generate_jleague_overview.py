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

PROMPT = """
今日のJリーグファンのSNSやニュースで盛り上がっている話題を検索し1つだけピックアップして、自然な文章にまとめて作成してください。
必ず完結した文章にしてください。「今日のJリーグのニュースでは」などのような前置きは絶対NG。

出力は次のJSON形式のみ: {"summary":"ここに文章"}
""".strip()
MAX_DUP_RETRIES = 2
SIMILARITY_THRESHOLD = 0.72


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


def _parse_summary(text: str) -> str:
    raw = text.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Allow fenced JSON, but do not accept truncated JSON.
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
                raw = "\n".join(lines[1:-1]).strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict) or "summary" not in data:
        raise ValueError("missing summary in JSON response")
    summary = data["summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("empty summary")
    return summary.strip()


def _is_complete_sentence(summary: str) -> bool:
    s = summary.strip()
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


def _list_models(api_key: str) -> List[dict[str, Any]]:
    url = f"{API_BASE}/{API_VERSION}/models"
    resp = requests.get(url, params={"key": api_key}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("models", [])
    return models if isinstance(models, list) else []


def _pick_model(models: List[dict[str, Any]]) -> str | None:
    def supports_generate_content(m: dict[str, Any]) -> bool:
        methods = m.get("supportedGenerationMethods", [])
        return "generateContent" in methods if isinstance(methods, list) else False

    eligible = [m for m in models if supports_generate_content(m)]
    if not eligible:
        return None

    def model_id(m: dict[str, Any]) -> str:
        name = m.get("name", "")
        return name.replace("models/", "")

    preferred = [m for m in eligible if model_id(m) == MODEL_NAME]
    if preferred:
        return model_id(preferred[0])
    if "gemini-2.5-flash" in [model_id(m) for m in eligible]:
        return "gemini-2.5-flash"
    flash = [m for m in eligible if "flash" in model_id(m)]
    if flash:
        return model_id(flash[0])
    return model_id(eligible[0])


def _build_prompt_with_recent(recent_summaries: List[str]) -> str:
    if not recent_summaries:
        return PROMPT
    lines = [f"- {s}" for s in recent_summaries]
    return (
        f"{PROMPT}\n\n"
        "直近に採用済みの概況（同じネタは避けること）:\n"
        + "\n".join(lines)
        + "\n\n同じ話題を繰り返さず、別の盛り上がりを1つ選んでください。"
    )


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


def _call_gemini(api_key: str, recent_summaries: List[str]) -> str:
    def build_payload(use_tools: bool, use_schema: bool, prompt_text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt_text}],
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1024,
            },
        }
        if use_tools:
            payload["tools"] = [{"google_search": {}}]
        if use_schema:
            payload["generationConfig"]["responseMimeType"] = "application/json"
            payload["generationConfig"]["responseSchema"] = {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            }
        return payload

    def post_with_timeout(endpoint: str, payload: dict[str, Any]) -> requests.Response:
        try:
            return requests.post(endpoint, params={"key": api_key}, json=payload, timeout=60)
        except requests.exceptions.ReadTimeout:
            return requests.post(endpoint, params={"key": api_key}, json=payload, timeout=120)

    selected_model = MODEL_NAME
    endpoint = _build_endpoint(selected_model, API_VERSION)
    recent = list(recent_summaries)
    last_error: Exception | None = None
    for _ in range(MAX_DUP_RETRIES + 1):
        prompt_text = _build_prompt_with_recent(recent[-5:])
        payload = build_payload(use_tools=True, use_schema=False, prompt_text=prompt_text)
        resp = post_with_timeout(endpoint, payload)
        if resp.status_code == 404:
            models = _list_models(api_key)
            picked = _pick_model(models)
            if picked and picked != selected_model:
                selected_model = picked
                endpoint = _build_endpoint(selected_model, API_VERSION)
                resp = post_with_timeout(endpoint, payload)
        if os.environ.get("GEMINI_DEBUG") == "1":
            print(f"debug: requested_model={MODEL_NAME} selected_model={selected_model}", file=sys.stderr)

        if resp.status_code >= 400:
            print(f"error: Gemini API status {resp.status_code}", file=sys.stderr)
            print(resp.text, file=sys.stderr)
            _log_error(f"Gemini API status {resp.status_code}", resp.text)
            resp.raise_for_status()

        data = resp.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected Gemini response format: {exc}")
        if os.environ.get("GEMINI_DEBUG") == "1":
            print("debug: raw gemini text:", repr(text), file=sys.stderr)
        try:
            summary = _parse_summary(text)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if os.environ.get("GEMINI_DEBUG") == "1":
                print("debug: invalid/empty JSON response:", repr(text), file=sys.stderr)
            continue
        if not _is_complete_sentence(summary):
            recent.append(summary)
            continue
        if not _is_similar_topic(summary, recent_summaries):
            return summary
        recent.append(summary)
    if last_error:
        raise ValueError(f"failed to parse valid summary after retries: {last_error}")
    raise ValueError("failed to generate non-duplicate complete summary after retries")


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
    cutoff = now - timedelta(hours=24)
    history = _prune_history(history, cutoff)
    recent_summaries = [h.get("summary_text", "") for h in history[-5:] if isinstance(h.get("summary_text"), str)]
    if isinstance(prev_current, dict) and isinstance(prev_current.get("summary_text"), str):
        recent_summaries.append(prev_current["summary_text"])

    summary = _call_gemini(api_key, recent_summaries)

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
