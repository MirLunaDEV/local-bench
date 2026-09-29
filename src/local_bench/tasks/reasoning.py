"""BBH, MuSR, and DROP."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from local_bench.scoreutil import short_answer, token_f1
from local_bench.tasks.base import Example, Task, mean_metrics, shot_count
from local_bench.tasks.hfutil import load_hf
from local_bench.tasks.mcq import _mmlu_block

_COT_DIR = Path(__file__).resolve().parents[1] / "third_party" / "bbh_cot"
_COT_PROMPTS: dict[str, tuple[str, list[str]]] | None = None


def cot_prompts() -> dict[str, tuple[str, list[str]]]:
    """Task name -> (instruction, three official CoT exemplars)."""
    global _COT_PROMPTS
    if _COT_PROMPTS is not None:
        return _COT_PROMPTS
    found: dict[str, tuple[str, list[str]]] = {}
    for path in sorted(_COT_DIR.glob("*.txt")):
        found[path.stem] = _split_cot(path.read_text(encoding="utf-8"))
    if len(found) != 27:
        raise RuntimeError(f"BBH CoT 프롬프트가 {len(found)}개입니다. 27개여야 합니다.")
    _COT_PROMPTS = found
    return found


def _split_cot(raw: str) -> tuple[str, list[str]]:
    text = raw.replace("\r\n", "\n")
    if "-----" in text:
        text = text.split("-----", 1)[1]
    text = text.strip("\n")
    parts = re.split(r"\n\n(?=Q:)", text)
    instruction = ""
    examples: list[str] = []
    for part in parts:
        piece = part.strip()
        if piece.startswith("Q:"):
            examples.append(piece)
        elif piece:
            instruction = piece
    return instruction, examples


def render_bbh_prompt(task_name: str, question: str, shots: int) -> str:
    """Official completion prompt: exemplars, then ``Q: ...\\nA: Let's think step by step.``"""
    question = question.strip()
    if question.lower().startswith("q:"):
        question = question[2:].strip()
    if shots <= 0:
        return question
    prompts = cot_prompts()
    if task_name not in prompts:
        raise RuntimeError(f"BBH CoT 프롬프트가 없습니다: {task_name}")
    instruction, examples = prompts[task_name]
    used = examples[:shots]
    chunks: list[str] = []
    if instruction:
        chunks.append(instruction)
    chunks.extend(used)
    chunks.append(f"Q: {question}\nA: Let's think step by step.")
    return "\n\n".join(chunks)


def _clip_bbh(value: str) -> str:
    value = value.strip().strip("`").strip().strip("\"'")
    value = value.split("\n")[0].strip().strip("`").strip()
    choice = re.match(r"^\(([A-Za-z])\)", value)
    if choice:
        return choice.group(0)
    value = re.split(r"\.\s+", value, maxsplit=1)[0]
    value = value.strip().rstrip(".").strip().strip("\"'")
    return re.sub(r"\s+", " ", value).strip()


def _norm_bbh(text: str) -> str:
    value = _clip_bbh(short_answer(text)).lower()
    if re.fullmatch(r"\([a-z]\)", value):
        value = value[1:-1]
    return value


def build_bbh(task: Task, opts) -> list[Example]:
    from datasets import get_dataset_config_names

    shots = max(0, shot_count(task, opts))
    names = get_dataset_config_names("lukaemon/bbh")
    if shots:
        missing = [name for name in names if name not in cot_prompts()]
        if missing:
            raise RuntimeError("BBH CoT 프롬프트가 없는 과제: " + ", ".join(missing))
    examples = []
    for name in names:
        rows = list(load_hf("lukaemon/bbh", name, "test"))
        for index, row in enumerate(rows):
            target = str(row.get("target") or row.get("answer") or "")
            question = str(row.get("input") or row.get("question") or "")
            examples.append(
                Example(
                    id=f"{name}-{index}",
                    messages=[{"role": "user", "content": render_bbh_prompt(name, question, shots)}],
                    gold=target,
                    meta={"category": name, "subject": name},
                )
            )
    return examples


def grade_bbh(example: Example, text: str) -> dict:
    pred = _norm_bbh(text)
    gold = _norm_bbh(str(example.gold))
    return {"score": 1.0 if pred == gold else 0.0, "pred": pred[:200]}


def _musr_choices(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    text = str(raw).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    raise ValueError("MuSR choices")


def build_musr(opts) -> list[Example]:
    del opts
    examples = []
    for name in ("murder_mysteries", "object_placements", "team_allocation"):
        rows = list(load_hf("TAUR-Lab/MuSR", None, name))
        for index, row in enumerate(rows):
            choices = _musr_choices(row.get("choices") or row.get("answer_choices"))
            letters = "ABCDEFGHIJ"
            pairs = [(letters[pos], choice) for pos, choice in enumerate(choices)]
            gold = letters[int(row["answer_index"])]
            narrative = str(row.get("narrative") or row.get("context") or "")
            question = str(row.get("question") or "")
            prompt = (
                "Read the story and answer the question. The last line must be 'Answer: X'.\n\n"
                f"{narrative.strip()}\n\n"
                + _mmlu_block(question, pairs, None)
            )
            examples.append(
                Example(
                    id=f"{name}-{index}",
                    messages=[{"role": "user", "content": prompt}],
                    gold=gold,
                    meta={"allowed": "".join(label for label, _ in pairs), "category": name, "subject": name},
                )
            )
    if not examples:
        raise RuntimeError("MuSR 문제를 읽지 못했습니다.")
    return examples


def build_drop(opts) -> list[Example]:
    del opts
    rows = list(load_hf("ucinlp/drop", None, "validation"))
    examples = []
    for index, row in enumerate(rows):
        spans = row.get("answers_spans") or {}
        if isinstance(spans, dict):
            golds = [str(item) for item in spans.get("spans") or []]
        else:
            golds = [str(spans)]
        prompt = (
            f"Passage:\n{str(row['passage']).strip()}\n\n"
            f"Question: {str(row['question']).strip()}\n\n"
            "Answer with only the final short answer on the last line, prefixed by 'Answer:'."
        )
        examples.append(
            Example(
                id=str(index),
                messages=[{"role": "user", "content": prompt}],
                gold=golds[0] if golds else "",
                meta={"golds": golds, "subject": "drop"},
            )
        )
    return examples


def grade_drop(example: Example, text: str) -> dict:
    pred = short_answer(text)
    golds = example.meta.get("golds") or [str(example.gold)]
    scores = [token_f1(pred, gold) for gold in golds] or [0.0]
    best = max(scores)
    exact = any(_norm_bbh(pred) == _norm_bbh(gold) for gold in golds)
    return {"score": best, "f1": best, "em": 1.0 if exact else 0.0, "pred": pred[:200]}


def aggregate_drop(rows: list[dict]) -> dict:
    summary = mean_metrics(rows)
    summary["score"] = summary.get("f1", summary.get("score"))
    return summary


def reasoning_tasks() -> list[Task]:
    from local_bench.tasks.hfutil import grade_letter

    bbh = Task(
        id="bbh",
        title="BIG-Bench Hard",
        group="reasoning",
        default_shots=3,
        protocol=(
            "Suzgun et al. 과제별 3-shot CoT. 프롬프트는 공개 cot-prompts 그대로이고, "
            "질문은 'Q:' 뒤에 붙인 뒤 'A: Let's think step by step.'으로 끝낸다. "
            "채점은 응답에서 마지막 'the answer is' 또는 'Answer:' 뒤를 정규화한 정확 일치. "
            "객관식 (B)와 B는 같은 답이다. --shots 0 은 예시 없는 문제만이라 논문 점수와 비교하지 않는다. "
            "logical deduction과 tracking shuffled objects의 3/5/7은 공개 파일이 같은 3-object 예시다."
        ),
        blurb="어려운 추론 과제 묶음. 산수, 논리, 날짜, 추적 문제.",
        grade_fn=grade_bbh,
    )
    bbh.build = lambda opts, task=bbh: build_bbh(task, opts)
    musr = Task(
        id="musr",
        title="MuSR",
        group="reasoning",
        protocol="서사 추론 객관식. murder / object / team 분할을 한 점수로 평균.",
        blurb="긴 이야기를 읽고 범인, 위치, 배정을 고른다.",
        build=build_musr,
        grade_fn=grade_letter,
    )
    drop = Task(
        id="drop",
        title="DROP",
        group="reading",
        metric="f1",
        protocol="DROP validation. 토큰 F1. 숫자 단어 정규화는 하지 않는다.",
        blurb="글을 읽고 숫자나 짧은 답을 뽑는 독해.",
        build=build_drop,
        grade_fn=grade_drop,
        aggregate_fn=aggregate_drop,
    )
    return [bbh, musr, drop]
