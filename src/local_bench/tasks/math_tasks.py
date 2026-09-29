"""GSM8K, MATH-500, AIME 2024, and MGSM."""

from __future__ import annotations

from local_bench.scoreutil import extract_boxed, extract_last_number, math_equal, numbers_equal
from local_bench.tasks.base import Example, Task, shot_count
from local_bench.tasks.hfutil import load_hf

_MGSM_LANGS = ("en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn")


def _gsm_prompt(shots: list[dict], question: str) -> str:
    parts = []
    for row in shots:
        parts.append(f"Q: {str(row['question']).strip()}\nA: {str(row['answer']).strip()}")
    parts.append(f"Q: {question.strip()}\nA:")
    return "\n\n".join(parts)


def _gold_gsm(answer: str) -> str:
    if "####" in answer:
        answer = answer.split("####")[-1]
    return answer.strip().replace(",", "")


def build_gsm8k(task: Task, opts) -> list[Example]:
    shots_n = shot_count(task, opts)
    test = list(load_hf("openai/gsm8k", "main", "test"))
    shots = list(load_hf("openai/gsm8k", "main", f"train[:{shots_n}]")) if shots_n else []
    examples = []
    for index, row in enumerate(test):
        examples.append(
            Example(
                id=str(index),
                messages=[{"role": "user", "content": _gsm_prompt(shots, str(row["question"]))} ],
                gold=_gold_gsm(str(row["answer"])),
                meta={"subject": "gsm8k"},
            )
        )
    return examples


def grade_number(example: Example, text: str) -> dict:
    boxed = extract_boxed(text)
    pred = extract_last_number(boxed) if boxed else None
    if pred is None:
        pred = extract_last_number(text)
    ok = numbers_equal(pred, str(example.gold))
    return {"score": 1.0 if ok else 0.0, "pred": pred}


def build_math500(task: Task, opts) -> list[Example]:
    del task, opts
    rows = list(load_hf("HuggingFaceH4/MATH-500", None, "test"))
    examples = []
    for index, row in enumerate(rows):
        problem = str(row.get("problem") or row.get("question"))
        prompt = "Solve the problem. Put the final answer in \\boxed{}.\n\n" + problem.strip()
        examples.append(
            Example(
                id=str(row.get("unique_id") or index),
                messages=[{"role": "user", "content": prompt}],
                gold=str(row.get("answer") or ""),
                meta={"subject": str(row.get("subject") or "math"), "category": str(row.get("subject") or "")},
            )
        )
    return examples


def grade_math(example: Example, text: str) -> dict:
    pred = extract_boxed(text) or ""
    if not pred:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        pred = lines[-1] if lines else ""
    ok = math_equal(pred, str(example.gold))
    return {"score": 1.0 if ok else 0.0, "pred": pred[:200]}


def build_aime(task: Task, opts) -> list[Example]:
    del task, opts
    rows = None
    errors = []
    for path, config, split in (
        ("Maxwell-Jia/AIME_2024", None, "train"),
        ("HuggingFaceH4/aime_2024", None, "train"),
        ("AI-MO/aimo-validation-aime", None, "train"),
    ):
        try:
            rows = list(load_hf(path, config, split))
            if rows:
                break
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            rows = None
    if not rows:
        raise RuntimeError("AIME 2024 데이터를 찾지 못했습니다. " + " | ".join(errors[:2]))
    examples = []
    for index, row in enumerate(rows):
        problem = str(row.get("Problem") or row.get("problem") or row.get("question"))
        answer = row.get("Answer", row.get("answer"))
        prompt = (
            "Solve this AIME problem. The answer is an integer from 0 to 999. "
            "Put the final answer in \\boxed{}.\n\n"
            + problem.strip()
        )
        examples.append(
            Example(
                id=str(row.get("ID") or row.get("id") or index),
                messages=[{"role": "user", "content": prompt}],
                gold=str(answer).strip(),
                meta={"subject": "aime2024"},
            )
        )
    return examples


def build_mgsm(task: Task, opts) -> list[Example]:
    del task
    want = (getattr(opts, "lang", None) or "").lower()
    languages = [want] if want else list(_MGSM_LANGS)
    examples = []
    for language in languages:
        rows = list(load_hf("juletxara/mgsm", language, "test"))
        for index, row in enumerate(rows):
            question = str(row.get("question") or row.get("Question"))
            answer = row.get("answer", row.get("answer_number"))
            prompt = f"Q: {question.strip()}\nA:"
            examples.append(
                Example(
                    id=f"{language}-{index}",
                    messages=[{"role": "user", "content": prompt}],
                    gold=str(answer).strip(),
                    meta={"subject": language, "category": language},
                )
            )
    return examples


def math_tasks() -> list[Task]:
    gsm = Task(
        id="gsm8k",
        title="GSM8K",
        group="math",
        default_shots=8,
        protocol="lm-eval gsm8k chain-of-thought, 8-shot. The answer is the number after ####. --shots 0 sends no examples.",
        blurb="1,319 grade-school word problems.",
    )
    gsm.build = lambda opts, task=gsm: build_gsm8k(task, opts)
    gsm.grade_fn = grade_number

    math500 = Task(
        id="math500",
        title="MATH-500",
        group="math",
        protocol=(
            "500 questions drawn from the MATH test. The text inside \\boxed{} is compared with the gold answer. "
            "They match when the normalized strings are equal, or when both sides parse as floats. "
            "\\frac{1}{2} and 0.5 count as different answers."
        ),
        blurb="500 contest-math questions. The final expression is read from a box.",
        build=lambda opts: build_math500(None, opts),
        grade_fn=grade_math,
    )
    aime = Task(
        id="aime2024",
        title="AIME 2024",
        group="math",
        protocol="30 AIME 2024 questions. Integers from 0 to 999.",
        blurb="30 questions from the 2024 American high-school math contest.",
        build=lambda opts: build_aime(None, opts),
        grade_fn=grade_number,
    )
    mgsm = Task(
        id="mgsm",
        title="MGSM",
        group="math",
        protocol="Multilingual GSM, 10 languages, 0-shot. --lang en runs one language.",
        blurb="GSM8K translated into 10 languages. Per-language scores are reported too.",
        build=lambda opts: build_mgsm(None, opts),
        grade_fn=grade_number,
    )
    return [gsm, math500, aime, mgsm]
