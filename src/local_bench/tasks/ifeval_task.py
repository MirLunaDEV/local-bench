"""IFEval prompt-level and instruction-level accuracy."""

from __future__ import annotations

import json

from local_bench.tasks.base import Example, Task, mean_metrics
from local_bench.tasks.hfutil import load_hf


def _variants(response: str) -> list[str]:
    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    pool = [
        response,
        response.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]
    seen = []
    for item in pool:
        if item not in seen:
            seen.append(item)
    return seen


def _kwargs_list(raw, count: int) -> list[dict]:
    if isinstance(raw, str):
        raw = json.loads(raw) if raw.strip() else []
    if isinstance(raw, dict):
        raw = [raw]
    items = list(raw or [])
    cleaned = []
    for item in items:
        if isinstance(item, str):
            item = json.loads(item) if item.strip() else {}
        if not isinstance(item, dict):
            item = {}
        cleaned.append({key: value for key, value in item.items() if value is not None})
    while len(cleaned) < count:
        cleaned.append({})
    return cleaned[:count]


def _follows(instruction_id: str, kwargs: dict, prompt: str, responses: list[str]) -> bool:
    from local_bench.third_party.instruction_following_eval.instructions_registry import (
        INSTRUCTION_DICT,
    )

    instruction = INSTRUCTION_DICT[instruction_id](instruction_id)
    keys = set(instruction.get_instruction_args_keys() or [])
    filtered = {key: value for key, value in kwargs.items() if not keys or key in keys}
    try:
        instruction.build_description(**filtered)
    except TypeError:
        instruction.build_description(**{key: filtered[key] for key in keys if key in filtered})
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt=prompt)
    for response in responses:
        if response.strip() and instruction.check_following(response):
            return True
    return False


def build_ifeval(opts) -> list[Example]:
    del opts
    rows = list(load_hf("google/IFEval", None, "train"))
    examples = []
    for index, row in enumerate(rows):
        ids = list(row["instruction_id_list"])
        kwargs = _kwargs_list(row.get("kwargs"), len(ids))
        examples.append(
            Example(
                id=str(row.get("key") if row.get("key") is not None else index),
                messages=[{"role": "user", "content": str(row["prompt"])}],
                gold=str(index),
                meta={"instruction_ids": ids, "kwargs": kwargs, "prompt": str(row["prompt"])},
            )
        )
    return examples


def grade_ifeval(example: Example, text: str) -> dict:
    ids = example.meta["instruction_ids"]
    kwargs = example.meta["kwargs"]
    prompt = example.meta["prompt"]
    strict = []
    loose = []
    for index, instruction_id in enumerate(ids):
        strict.append(_follows(instruction_id, kwargs[index], prompt, [text]))
        loose.append(_follows(instruction_id, kwargs[index], prompt, _variants(text)))
    total = len(ids) or 1
    prompt_strict = 1.0 if ids and all(strict) else 0.0
    prompt_loose = 1.0 if ids and all(loose) else 0.0
    return {
        "score": prompt_strict,
        "prompt_strict": prompt_strict,
        "prompt_loose": prompt_loose,
        "inst_strict": sum(strict) / total,
        "inst_loose": sum(loose) / total,
        "n_inst": float(len(ids)),
        "n_strict": float(sum(strict)),
        "n_loose": float(sum(loose)),
    }


def aggregate_ifeval(rows: list[dict]) -> dict:
    summary = mean_metrics(rows)
    scored = [row for row in rows if not row.get("error")]
    inst = sum(float((row.get("metrics") or {}).get("n_inst") or 0) for row in scored)
    strict = sum(float((row.get("metrics") or {}).get("n_strict") or 0) for row in scored)
    loose = sum(float((row.get("metrics") or {}).get("n_loose") or 0) for row in scored)
    if inst:
        summary["instruction_strict"] = strict / inst
        summary["instruction_loose"] = loose / inst
    summary["score"] = summary.get("prompt_strict", summary.get("score"))
    return summary


def ifeval_task() -> Task:
    return Task(
        id="ifeval",
        title="IFEval",
        group="instruction",
        metric="prompt_strict",
        protocol="Google IFEval. 대표 점수는 prompt-level strict. instruction-level strict/loose도 같이 계산한다. 문장 수는 NLTK punkt 대신 정규식 분할이다.",
        blurb="형식 지시를 지켰는지 자동으로 확인한다. 쉼표 금지, JSON, 글자 수, 키워드.",
        build=build_ifeval,
        grade_fn=grade_ifeval,
        aggregate_fn=aggregate_ifeval,
    )
