"""Run examples against a chat client and store JSONL plus a summary."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from local_bench.scoreutil import visible_text
from local_bench.tasks.base import Example, mean_metrics


def slug(text: str) -> str:
    cleaned = []
    for char in text:
        if char.isalnum() or char in "._-":
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "model"


class JsonlStore:
    def __init__(self, path: Path):
        self.path = path
        self.rows: dict[str, dict] = {}
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        kept: list[str] = []
        dropped_tail = False
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A crash mid-append leaves a partial last line. Drop only that
                # tail so the next run can resume. A broken line in the middle
                # is a different kind of corruption and still raises.
                if index == len(lines) - 1:
                    dropped_tail = True
                    continue
                raise
            self.rows[str(row["id"])] = row
            kept.append(line)
        if dropped_tail:
            payload = ("\n".join(kept) + "\n") if kept else ""
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(path)
            print(
                f"이어하기: {path.name} 의 마지막 줄이 잘려 있어 버렸습니다. 그 문항은 다시 받습니다.",
                file=sys.stderr,
            )

    def get(self, item_id: str) -> dict | None:
        return self.rows.get(item_id)

    def add(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows[str(row["id"])] = row
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _json_gold(example: Example):
    if isinstance(example.gold, (str, int, float, bool)) or example.gold is None:
        return example.gold
    return example.meta.get("gold_display")


def _filter_examples(task, examples: list[Example], opts) -> list[Example]:
    subject = getattr(opts, "subject", None)
    if subject:
        want = subject.replace(" ", "_").lower()
        matched = []
        for example in examples:
            have = str(example.meta.get("subject") or "").replace(" ", "_").lower()
            if have == want or (want and want in have):
                matched.append(example)
        if not matched:
            known = sorted({str(example.meta.get("subject")) for example in examples if example.meta.get("subject")})
            preview = ", ".join(known[:25])
            raise RuntimeError(f"{task.id}: subject '{subject}' 에 해당하는 문제가 없습니다. 예: {preview}")
        examples = matched
    limit = getattr(opts, "limit", None)
    if limit is not None:
        examples = examples[: int(limit)]
    return examples


def run_task(task, client, opts, runs_dir: Path) -> dict:
    if hasattr(task, "execute"):
        return task.execute(client, opts, runs_dir)
    print(f"데이터 준비: {task.title} ({task.id})", file=sys.stderr)
    examples = _filter_examples(task, task.examples(opts), opts)
    if not examples:
        raise RuntimeError(f"{task.id}: 채점할 문제가 없습니다.")
    if getattr(opts, "limit", None):
        print(
            f"{task.id}: 앞에서 {len(examples)}개만 채점합니다. 리더보드 숫자와 비교하려면 --limit 없이 실행하세요.",
            file=sys.stderr,
        )
    path = runs_dir / slug(client.model) / f"{task.id}.jsonl"
    if getattr(opts, "fresh", False) and path.exists():
        path.unlink()
    store = JsonlStore(path)
    pending = [example for example in examples if store.get(example.id) is None]
    print(f"{task.id}: {len(examples) - len(pending)}개 이어서, {len(pending)}개 생성", file=sys.stderr)
    workers = max(1, int(getattr(opts, "workers", 1) or 1))
    consecutive = 0
    max_errors = int(getattr(opts, "max_errors", 8) or 8)
    done = len(examples) - len(pending)

    def work(example: Example) -> dict:
        started = time.perf_counter()
        usage = None
        try:
            if hasattr(client, "complete"):
                reply = client.complete(example.messages)
                raw = reply.text
                usage = getattr(reply, "usage", None)
            else:
                raw = client.chat(example.messages)
            text = visible_text(raw)
            metrics = task.grade(example, text)
            error = None
        except Exception as exc:  # transport and grader failures stay in the row
            raw = ""
            metrics = {}
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "id": example.id,
            "gold": _json_gold(example),
            "response": raw[:100000],
            "metrics": metrics,
            "latency_s": round(time.perf_counter() - started, 3),
            "error": error,
            "category": example.meta.get("category"),
        }
        kept_usage = _usage_counts(usage)
        if kept_usage:
            row["usage"] = kept_usage
        return row

    try:
        if workers == 1:
            for example in pending:
                row = work(example)
                store.add(row)
                done += 1
                consecutive = _report_progress(task.id, done, len(examples), row, consecutive, max_errors)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(work, example) for example in pending]
                for future in as_completed(futures):
                    row = future.result()
                    store.add(row)
                    done += 1
                    consecutive = _report_progress(task.id, done, len(examples), row, consecutive, max_errors)
    except KeyboardInterrupt:
        summary = _write_summary(task, client, opts, path, store, examples)
        print(f"\n중단됨. 같은 명령을 다시 실행하면 {path} 에서 이어갑니다.", file=sys.stderr)
        return summary
    except RuntimeError:
        if store.rows:
            _write_summary(task, client, opts, path, store, examples)
        raise
    return _write_summary(task, client, opts, path, store, examples)


def _report_progress(task_id: str, done: int, total: int, row: dict, consecutive: int, max_errors: int) -> int:
    if row.get("error"):
        consecutive += 1
        status = f"error {row['error'][:160]}"
    else:
        consecutive = 0
        score = (row.get("metrics") or {}).get("score")
        status = f"score={score}"
    print(f"[{done}/{total}] {task_id} {row['id']} {status}", file=sys.stderr)
    if consecutive >= max_errors:
        raise RuntimeError(f"{task_id}: 연속 {consecutive}회 호출 실패. 서버와 모델 id를 확인하세요. 마지막 오류: {row.get('error')}")
    return consecutive


def _usage_counts(usage) -> dict | None:
    if not isinstance(usage, dict):
        return None
    kept = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        kept[key] = int(value)
    return kept or None


def _ordered_rows(store: JsonlStore, examples: list[Example] | None) -> list[dict]:
    if not examples:
        return [store.rows[key] for key in store.rows]
    ordered = []
    seen: set[str] = set()
    for example in examples:
        row = store.rows.get(example.id)
        if row is None:
            continue
        ordered.append(row)
        seen.add(str(example.id))
    for key, row in store.rows.items():
        if key not in seen:
            ordered.append(row)
    return ordered


def _completion_tokens_per_s(rows: list[dict]) -> float | None:
    tokens = 0.0
    seconds = 0.0
    for row in rows:
        usage = row.get("usage") or {}
        count = usage.get("completion_tokens")
        latency = row.get("latency_s") or 0
        if isinstance(count, bool) or not isinstance(count, (int, float)):
            continue
        if not isinstance(latency, (int, float)) or isinstance(latency, bool) or latency <= 0:
            continue
        tokens += float(count)
        seconds += float(latency)
    if tokens <= 0 or seconds <= 0:
        return None
    return round(tokens / seconds, 2)


def _write_summary(task, client, opts, path: Path, store: JsonlStore, examples: list[Example] | None = None) -> dict:
    ordered = _ordered_rows(store, examples)
    metrics = task.aggregate(ordered)
    summary = {
        "task": task.id,
        "title": task.title,
        "model": client.model,
        "base_url": client.base_url,
        "protocol": task.protocol,
        "primary_metric": task.metric,
        "metrics": metrics,
        "limit": getattr(opts, "limit", None),
        "shots": None if getattr(opts, "shots", None) is None else int(opts.shots),
        "path": str(path),
    }
    rate = _completion_tokens_per_s(ordered)
    if rate is not None:
        summary["completion_tokens_per_s"] = rate
    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def format_summary(summary: dict) -> str:
    metrics = summary.get("metrics") or {}
    primary = summary.get("primary_metric") or "score"
    value = metrics.get(primary, metrics.get("score"))
    if summary.get("task") == "arena-hard" and isinstance(value, float):
        shown = f"{value:.3f} (0.5면 기준 모델과 동점)"
    elif isinstance(value, float):
        shown = f"{value * 100:.1f}%"
    else:
        shown = str(value)
    lines = [
        f"{summary.get('title') or summary.get('task')}  {shown}  (n={metrics.get('n')}, errors={metrics.get('errors')})",
        f"  모델 {summary.get('model')}  @ {summary.get('base_url')}",
        f"  프로토콜 {summary.get('protocol')}",
    ]
    rate = summary.get("completion_tokens_per_s")
    if isinstance(rate, (int, float)) and not isinstance(rate, bool):
        lines.append(f"  생성 {rate:.1f} tok/s")
    extras = []
    for key, item in metrics.items():
        if key in {"n", "errors", "by_category", primary, "score"}:
            continue
        if isinstance(item, float):
            extras.append(f"{key}={item * 100:.1f}%" if item <= 1 else f"{key}={item:.3f}")
    if extras:
        lines.append("  " + "  ".join(extras))
    by_category = metrics.get("by_category") or {}
    if by_category:
        parts = [f"{name} {row['score'] * 100:.1f}% (n={row['n']})" for name, row in by_category.items()]
        lines.append("  범주 " + ", ".join(parts))
    return "\n".join(lines)


def load_summaries(runs_dir: Path, model: str | None = None) -> list[dict]:
    summaries = []
    if not runs_dir.exists():
        return summaries
    paths = sorted(runs_dir.glob("*/*.summary.json"))
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if model and model.lower() not in str(row.get("model", "")).lower():
            continue
        summaries.append(row)
    return summaries


def fallback_mean(rows: list[dict]) -> dict:
    return mean_metrics(rows)
