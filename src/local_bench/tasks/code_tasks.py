"""HumanEval pass@1 and MBPP pass@1."""

from __future__ import annotations

import gzip
import json
import re

from local_bench.download import cached_download
from local_bench.sandbox import run_python
from local_bench.tasks.base import Example, Task

_HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
_FENCE = re.compile(r"```(?:python|py)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_code(text: str) -> str:
    fences = [chunk.strip() for chunk in _FENCE.findall(text or "") if chunk.strip()]
    for chunk in reversed(fences):
        if "def " in chunk or "class " in chunk:
            return chunk
    if fences:
        return fences[-1]
    if "def " in (text or ""):
        return text[text.find("def ") :].strip()
    return (text or "").strip()


def _humaneval_program(problem: dict, completion: str) -> str:
    entry = problem["entry_point"]
    if re.search(rf"(?m)^\s*def\s+{re.escape(entry)}\b", completion):
        body = completion
    else:
        body = problem["prompt"] + completion
        if not body.endswith("\n"):
            body += "\n"
    return f"{body}\n{problem['test']}\ncheck({entry})\n"


def build_humaneval(opts) -> list[Example]:
    del opts
    path = cached_download(_HUMANEVAL_URL, "HumanEval.jsonl.gz")
    examples = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            problem = json.loads(line)
            prompt = (
                "Write a Python function that satisfies the specification. "
                "Reply with one python code block containing the complete function.\n\n"
                f"{problem['prompt']}"
            )
            examples.append(
                Example(
                    id=problem["task_id"],
                    messages=[{"role": "user", "content": prompt}],
                    gold=problem["task_id"],
                    meta={"problem": problem, "timeout": 5.0, "kind": "humaneval"},
                )
            )
    return examples


def build_mbpp(opts) -> list[Example]:
    del opts
    from datasets import load_dataset

    dataset = load_dataset("google-research-datasets/mbpp", "full", split="test")
    examples = []
    for row in dataset:
        setup = str(row.get("test_setup_code") or "").strip()
        tests = (setup + "\n") if setup else ""
        tests += "\n".join(row.get("test_imports") or [])
        if row.get("test_imports"):
            tests += "\n"
        tests += "\n".join(row["test_list"])
        prompt = (
            "You are an expert Python programmer, and here is your task:\n"
            f"{row['text'].strip()}\n\n"
            "Your code should pass these tests:\n"
            f"{tests}\n\n"
            "Reply with one python code block."
        )
        task_id = str(row.get("task_id"))
        examples.append(
            Example(
                id=task_id,
                messages=[{"role": "user", "content": prompt}],
                gold=task_id,
                meta={"tests": tests, "timeout": 5.0, "kind": "mbpp"},
            )
        )
    return examples


def grade_code(example: Example, text: str) -> dict:
    code = extract_code(text)
    kind = example.meta.get("kind")
    if kind == "humaneval":
        program = _humaneval_program(example.meta["problem"], code)
    else:
        program = code + "\n\n" + example.meta["tests"]
    passed, detail = run_python(program, timeout=float(example.meta.get("timeout") or 5))
    return {"score": 1.0 if passed else 0.0, "pred": code[:500], "detail": detail[:300]}


def code_tasks() -> list[Task]:
    humaneval = Task(
        id="humaneval",
        title="HumanEval",
        group="code",
        protocol=(
            "OpenAI HumanEval, 164 questions, pass@1. The chat model writes the whole function, which is run against the official tests. "
            "Only a subprocess and a time limit are used. This is not a security sandbox."
        ),
        blurb="164 Python functions. The model's code runs on this machine and scores correct when the tests pass.",
        build=build_humaneval,
        grade_fn=grade_code,
    )
    mbpp = Task(
        id="mbpp",
        title="MBPP",
        group="code",
        protocol=(
            "MBPP full test, 500 questions. The prompt includes the problem and the public tests, and the score is pass@1. "
            "Only a subprocess and a time limit are used. This is not a security sandbox."
        ),
        blurb="500 short Python problems. Easier than HumanEval, and the tests are included in the prompt.",
        build=build_mbpp,
        grade_fn=grade_code,
    )
    return [humaneval, mbpp]
