"""BFCL v4 Python prompting track: simple, multiple, parallel, and irrelevance."""

from __future__ import annotations

import json

from local_bench.download import cached_download, load_json_records
from local_bench.tasks.base import Example, Task, mean_metrics
from local_bench.tasks.bfcl_decode import decode_calls

_DATA = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data/"
_ANSWER = _DATA + "possible_answer/"

CATEGORIES = {
    "simple_python": "BFCL_v4_simple_python.json",
    "multiple": "BFCL_v4_multiple.json",
    "parallel": "BFCL_v4_parallel.json",
    "parallel_multiple": "BFCL_v4_parallel_multiple.json",
    "live_simple": "BFCL_v4_live_simple.json",
    "live_multiple": "BFCL_v4_live_multiple.json",
    "live_parallel": "BFCL_v4_live_parallel.json",
    "live_parallel_multiple": "BFCL_v4_live_parallel_multiple.json",
    "irrelevance": "BFCL_v4_irrelevance.json",
}

_DEFAULT = ("simple_python", "multiple", "parallel", "parallel_multiple")

_SYSTEM = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in JSON format that you can invoke.
{functions}
"""


def _read_records(path) -> list[dict]:
    return load_json_records(path.read_text(encoding="utf-8"))


def _question_text(question) -> str:
    chunks = []

    def walk(node) -> None:
        if isinstance(node, dict):
            if node.get("role") == "user":
                chunks.append(str(node.get("content") or ""))
            return
        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(question)
    return "\n".join(chunks).strip()


def _selected(opts) -> list[str]:
    raw = getattr(opts, "category", None) or ""
    if not raw or raw == "default":
        return list(_DEFAULT)
    if raw == "all":
        return list(CATEGORIES)
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in names if name not in CATEGORIES]
    if unknown:
        known = ", ".join(CATEGORIES)
        raise RuntimeError(f"알 수 없는 BFCL 범주: {', '.join(unknown)}. 사용 가능: {known}")
    return names


def build_bfcl(opts) -> list[Example]:
    examples = []
    for category in _selected(opts):
        filename = CATEGORIES[category]
        questions = _read_records(cached_download(_DATA + filename, "bfcl/" + filename))
        answers = {}
        if category != "irrelevance":
            answer_rows = _read_records(cached_download(_ANSWER + filename, "bfcl/answer/" + filename))
            answers = {row["id"]: row["ground_truth"] for row in answer_rows}
        for row in questions:
            functions = row.get("function") or []
            if isinstance(functions, dict):
                functions = [functions]
            user = _question_text(row.get("question"))
            messages = [
                {"role": "system", "content": _SYSTEM.format(functions=json.dumps(functions, ensure_ascii=False))},
                {"role": "user", "content": user},
            ]
            names = []
            ground = answers.get(row["id"])
            if isinstance(ground, list):
                for item in ground:
                    if isinstance(item, dict) and item:
                        names.append(next(iter(item)))
            examples.append(
                Example(
                    id=f"{category}:{row['id']}",
                    messages=messages,
                    gold=ground if ground is not None else [],
                    meta={
                        "category": category,
                        "functions": functions,
                        "gold_display": ",".join(names) if names else category,
                        "irrelevance": category == "irrelevance",
                    },
                )
            )
    return examples


def grade_bfcl(example: Example, text: str) -> dict:
    from local_bench.third_party.bfcl.ast_checker import Language, ast_checker

    functions = example.meta["functions"]
    calls = decode_calls(text, functions)
    if example.meta.get("irrelevance"):
        ok = not calls
        return {"score": 1.0 if ok else 0.0, "pred": _preview(calls)}
    if not calls:
        return {"score": 0.0, "pred": None, "detail": "call을 해석하지 못했습니다."}
    category = str(example.meta["category"])
    # parallel_multiple contains both words. The official checker tests parallel first.
    test_category = category
    result = ast_checker(
        functions,
        calls,
        example.gold,
        Language.PYTHON,
        test_category,
        "local",
    )
    return {
        "score": 1.0 if result.get("valid") else 0.0,
        "pred": _preview(calls),
        "detail": "" if result.get("valid") else str(result.get("error_type") or "")[:180],
    }


def _preview(calls) -> str:
    if not calls:
        return ""
    try:
        return json.dumps(calls, ensure_ascii=False)[:400]
    except TypeError:
        return str(calls)[:400]


def aggregate_bfcl(rows: list[dict]) -> dict:
    summary = mean_metrics(rows)
    categories = summary.get("by_category") or {}
    if categories:
        summary["score"] = sum(item["score"] for item in categories.values()) / len(categories)
        summary["macro_score"] = summary["score"]
    return summary


def bfcl_task() -> Task:
    return Task(
        id="bfcl",
        title="BFCL",
        group="agent",
        protocol=(
            "Berkeley Function Calling Leaderboard v4 파이썬 prompting. "
            "기본은 simple, multiple, parallel, parallel_multiple의 AST 정확도 매크로 평균. "
            "live와 irrelevance는 --category 로 켠다. 멀티턴 실행 환경은 포함하지 않는다."
        ),
        blurb="함수 호출 에이전트 벤치. 이름, 인자, 값이 정답 집합과 맞는지 본다.",
        build=build_bfcl,
        grade_fn=grade_bfcl,
        aggregate_fn=aggregate_bfcl,
    )
