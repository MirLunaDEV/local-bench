"""Shared task types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Example:
    id: str
    messages: list[dict]
    gold: object
    meta: dict = field(default_factory=dict)


@dataclass
class Task:
    id: str
    title: str
    group: str
    protocol: str
    metric: str = "score"
    blurb: str = ""
    default_shots: int = 0
    build: Callable | None = None
    grade_fn: Callable | None = None
    aggregate_fn: Callable | None = None

    def examples(self, opts) -> list[Example]:
        if self.build is None:
            raise NotImplementedError(self.id)
        return self.build(opts)

    def grade(self, example: Example, text: str) -> dict:
        if self.grade_fn is None:
            raise NotImplementedError(self.id)
        return self.grade_fn(example, text)

    def aggregate(self, rows: list[dict]) -> dict:
        if self.aggregate_fn is not None:
            return self.aggregate_fn(rows)
        return mean_metrics(rows)


def shot_count(task: Task, opts) -> int:
    if getattr(opts, "shots", None) is None:
        return task.default_shots
    return int(opts.shots)


def mean_metrics(rows: list[dict]) -> dict:
    scored = [row for row in rows if not row.get("error")]
    keys: list[str] = []
    for row in scored:
        for key, value in (row.get("metrics") or {}).items():
            if key.startswith("n_"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if key not in keys:
                keys.append(key)
    summary: dict = {
        "n": len(scored),
        "errors": sum(1 for row in rows if row.get("error")),
    }
    for key in keys:
        values = []
        for row in scored:
            value = (row.get("metrics") or {}).get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values.append(float(value))
        if values:
            summary[key] = sum(values) / len(values)
    categories: dict[str, list[float]] = {}
    for row in scored:
        category = row.get("category")
        score = (row.get("metrics") or {}).get("score")
        if category and isinstance(score, (int, float)) and not isinstance(score, bool):
            categories.setdefault(str(category), []).append(float(score))
    if categories:
        summary["by_category"] = {
            name: {"n": len(values), "score": sum(values) / len(values)}
            for name, values in sorted(categories.items())
        }
    return summary
