"""Cached HTTP downloads and slightly dirty JSONL files."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = "local-bench"


def cache_dir() -> Path:
    override = os.environ.get("LOCAL_BENCH_CACHE")
    path = Path(override) if override else Path.home() / ".cache" / "local-bench"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cached_download(url: str, name: str, timeout: float = 120) -> Path:
    dest = cache_dir() / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                tmp.write_bytes(response.read())
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"다운로드 실패: {url} ({last_error})") from last_error


def load_json_records(text: str) -> list:
    """Read a JSON array or JSONL.

    Arena-Hard has two prompts with a raw newline inside a JSON string.
    Those physical lines are joined until the record parses.
    """
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[0] == "[":
        data = json.loads(stripped)
        return data if isinstance(data, list) else [data]
    rows = []
    buffer = ""
    for line in text.splitlines():
        if not line and not buffer:
            continue
        buffer = line if not buffer else buffer + "\\n" + line
        try:
            rows.append(json.loads(buffer))
        except json.JSONDecodeError:
            continue
        buffer = ""
    if buffer.strip():
        raise ValueError("JSONL 레코드가 끝나지 않았습니다.")
    return rows
