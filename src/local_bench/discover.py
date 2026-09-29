"""Find local OpenAI-compatible servers and their model ids."""

from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from local_bench.client import normalize_base
from local_bench.download import USER_AGENT

# name, OpenAI base, optional native model list (Ollama).
_PROBES = (
    ("Ollama", "http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/api/tags"),
    ("LM Studio", "http://127.0.0.1:1234/v1", None),
    ("llama.cpp", "http://127.0.0.1:8080/v1", None),
    ("vLLM / SGLang", "http://127.0.0.1:8000/v1", None),
    ("Q38", "http://127.0.0.1:1923/v1", None),
    ("LocalAI / other", "http://127.0.0.1:5000/v1", None),
)


def list_model_ids(base_url: str, api_key: str = "local", timeout: float = 2.0) -> list[str]:
    base = normalize_base(base_url)
    request = urllib.request.Request(
        base + "/models",
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("data") or payload.get("models") or []
    ids: list[str] = []
    for row in rows:
        if isinstance(row, str):
            ids.append(row)
        elif isinstance(row, dict):
            name = row.get("id") or row.get("name")
            if name:
                ids.append(str(name))
    return ids


def _ollama_names(url: str, timeout: float) -> list[str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [str(row["name"]) for row in payload.get("models") or [] if row.get("name")]


def _probe(spec: tuple[str, str, str | None], timeout: float, api_key: str) -> list[dict]:
    backend, base, native = spec
    found: list[dict] = []
    try:
        for model in list_model_ids(base, api_key=api_key, timeout=timeout):
            found.append({"backend": backend, "base_url": normalize_base(base), "model": model})
    except (OSError, json.JSONDecodeError, UnicodeError, ValueError):
        found = []
    if found:
        return found
    if native:
        try:
            for model in _ollama_names(native, timeout):
                found.append({"backend": backend, "base_url": normalize_base(base), "model": model})
        except (OSError, json.JSONDecodeError, UnicodeError, ValueError, KeyError):
            return []
    return found


def discover(api_key: str = "local", timeout: float = 1.5) -> list[dict]:
    specs = list(_PROBES)
    extra = os.environ.get("LOCAL_BENCH_BASE_URL")
    if extra:
        specs.insert(0, ("LOCAL_BENCH_BASE_URL", extra, None))
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(specs) or 1) as pool:
        futures = [pool.submit(_probe, spec, timeout, api_key) for spec in specs]
        for future in futures:
            rows.extend(future.result())
    unique: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["base_url"], row["model"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def resolve_endpoint(
    model: str | None,
    base_url: str | None,
    api_key: str,
    interactive: bool,
) -> tuple[str, str]:
    """Return (base_url, model_id). Asks on a terminal when several models are up."""
    if base_url and model:
        return normalize_base(base_url), model
    if base_url and not model:
        ids = list_model_ids(base_url, api_key=api_key, timeout=5)
        if not ids:
            raise SystemExit(f"{base_url} 에서 모델을 찾지 못했습니다.")
        if len(ids) == 1:
            return normalize_base(base_url), ids[0]
        picked = _choose([{"backend": "지정한 서버", "base_url": normalize_base(base_url), "model": mid} for mid in ids], interactive)
        return picked["base_url"], picked["model"]

    found = discover(api_key=api_key)
    if model:
        exact = [row for row in found if row["model"] == model]
        partial = exact or [row for row in found if model.lower() in row["model"].lower()]
        if len(partial) == 1:
            return partial[0]["base_url"], partial[0]["model"]
        if len(partial) > 1:
            lines = "\n".join(f"  {row['model']}  {row['base_url']}" for row in partial)
            raise SystemExit(f"모델 '{model}' 이 여러 서버에 있습니다. --base-url 을 지정하세요.\n{lines}")
        known = "\n".join(f"  {row['model']}  ({row['backend']} {row['base_url']})" for row in found) or "  (실행 중인 서버 없음)"
        raise SystemExit(f"모델 '{model}' 을 찾지 못했습니다.\n{known}")

    if len(found) == 1:
        return found[0]["base_url"], found[0]["model"]
    if not found:
        raise SystemExit(
            "실행 중인 로컬 서버를 찾지 못했습니다. Ollama, LM Studio, llama.cpp, vLLM, Q38(1923) 중 하나를 켜거나 --base-url 을 주세요."
        )
    picked = _choose(found, interactive)
    return picked["base_url"], picked["model"]


def _choose(rows: list[dict], interactive: bool) -> dict:
    print("로컬 모델:")
    for index, row in enumerate(rows, start=1):
        print(f"  {index}. {row['model']}   [{row['backend']}] {row['base_url']}")
    if not interactive:
        raise SystemExit("모델이 여러 개입니다. --model 과 필요하면 --base-url 을 지정하세요.")
    raw = input("번호: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(rows):
        return rows[int(raw) - 1]
    for row in rows:
        if row["model"] == raw:
            return row
    raise SystemExit("목록에 없는 선택입니다.")
