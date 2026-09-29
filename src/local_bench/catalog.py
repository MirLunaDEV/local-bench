"""Task registry and named suites."""

from __future__ import annotations

from local_bench.tasks.arena import arena_task
from local_bench.tasks.bfcl import bfcl_task
from local_bench.tasks.code_tasks import code_tasks
from local_bench.tasks.gpqa import gpqa_tasks
from local_bench.tasks.ifeval_task import ifeval_task
from local_bench.tasks.math_tasks import math_tasks
from local_bench.tasks.mcq import knowledge_tasks
from local_bench.tasks.reasoning import reasoning_tasks

SUITES: dict[str, list[str]] = {
    "quick": ["gpqa", "humaneval", "aime2024", "openbookqa", "ifeval"],
    "core": ["gpqa", "mmlu", "humaneval", "gsm8k", "arc-challenge", "ifeval", "bfcl"],
    "knowledge": [
        "mmlu",
        "mmlu-pro",
        "kmmlu",
        "arc-challenge",
        "arc-easy",
        "openbookqa",
        "commonsenseqa",
        "boolq",
        "winogrande",
        "truthfulqa",
        "siqa",
        "hellaswag",
        "piqa",
    ],
    "science": ["gpqa", "gpqa-main", "gpqa-extended", "arc-challenge", "openbookqa"],
    "math": ["gsm8k", "math500", "aime2024", "mgsm"],
    "code": ["humaneval", "mbpp"],
    "instruction": ["ifeval"],
    "reasoning": ["gpqa", "bbh", "musr", "gsm8k", "math500"],
    "reading": ["boolq", "drop", "siqa"],
    "agent": ["bfcl"],
    "arena": ["arena-hard"],
    "aa-public": ["gpqa", "mmlu-pro", "gsm8k", "math500", "aime2024", "humaneval", "ifeval", "bfcl"],
}


def all_tasks() -> list:
    tasks = []
    tasks.extend(gpqa_tasks())
    tasks.extend(knowledge_tasks())
    tasks.extend(math_tasks())
    tasks.extend(code_tasks())
    tasks.append(ifeval_task())
    tasks.extend(reasoning_tasks())
    tasks.append(bfcl_task())
    tasks.append(arena_task())
    return tasks


def task_map() -> dict:
    return {task.id: task for task in all_tasks()}


def suite_ids(name: str) -> list[str]:
    if name == "all":
        return [task.id for task in all_tasks() if task.id != "arena-hard"]
    if name not in SUITES:
        known = ", ".join(["all", *SUITES])
        raise SystemExit(f"없는 스위트 '{name}'. 사용 가능: {known}")
    return list(SUITES[name])


def resolve_targets(spec: str) -> list:
    tasks = task_map()
    chosen = []
    seen: set[str] = set()
    for token in spec.split(","):
        name = token.strip()
        if not name:
            continue
        if name.startswith("suite:"):
            name = name.split(":", 1)[1]
        if name in SUITES or name == "all":
            ids = suite_ids(name)
        elif name in tasks:
            ids = [name]
        else:
            known = ", ".join(sorted(tasks))
            raise SystemExit(f"없는 벤치 '{token.strip()}'.\n태스크: {known}\n스위트: all, {', '.join(SUITES)}")
        for item_id in ids:
            if item_id in seen:
                continue
            seen.add(item_id)
            chosen.append(tasks[item_id])
    if not chosen:
        raise SystemExit("실행할 벤치가 없습니다.")
    return chosen
