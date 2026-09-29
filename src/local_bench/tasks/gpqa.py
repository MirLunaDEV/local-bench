"""GPQA, scored with the OpenAI simple-evals prompt."""

from __future__ import annotations

import csv
import io
import random
import zipfile

from local_bench.download import cached_download
from local_bench.scoreutil import extract_letter
from local_bench.tasks.base import Example, Task

_TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}""".strip()

_ZIP_URL = "https://github.com/idavidrein/gpqa/raw/main/dataset.zip"
# Published in the GPQA repository README so the dataset can be downloaded.
_ZIP_PASSWORD = b"deserted-untie-orchid"
_FILES = {
    "gpqa": "dataset/gpqa_diamond.csv",
    "gpqa-main": "dataset/gpqa_main.csv",
    "gpqa-extended": "dataset/gpqa_extended.csv",
}

_BLURBS = {
    "gpqa": "198 graduate-science questions. The Diamond set Artificial Analysis kept publishing on its own after removing it from the index.",
    "gpqa-main": "448 GPQA main questions. Broader than Diamond, and scored with the same prompt.",
    "gpqa-extended": "GPQA extended. Main plus questions whose checks were looser.",
}


def _load_rows(kind: str) -> list[dict]:
    path = cached_download(_ZIP_URL, "gpqa-dataset.zip")
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(_FILES[kind], pwd=_ZIP_PASSWORD)
    text = raw.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _build(kind: str, opts) -> list[Example]:
    rows = _load_rows(kind)
    repeats = max(1, int(getattr(opts, "repeats", 1) or 1))
    rng = random.Random(0)
    examples: list[Example] = []
    for copy in range(repeats):
        for index, row in enumerate(rows):
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            order = rng.sample(range(4), 4)
            choices = [choices[pos] for pos in order]
            gold = "ABCD"[choices.index(row["Correct Answer"])]
            prompt = _TEMPLATE.format(
                Question=row["Question"],
                A=choices[0],
                B=choices[1],
                C=choices[2],
                D=choices[3],
            )
            subject = row.get("High-level domain") or row.get("Subdomain") or ""
            item_id = str(index) if repeats == 1 else f"{index}#{copy}"
            examples.append(
                Example(
                    id=item_id,
                    messages=[{"role": "user", "content": prompt}],
                    gold=gold,
                    meta={"subject": subject, "allowed": "ABCD", "category": subject or None},
                )
            )
    return examples


def grade_gpqa(example: Example, text: str) -> dict:
    pred = extract_letter(text, "ABCD")
    return {"score": 1.0 if pred == example.gold else 0.0, "pred": pred}


def gpqa_tasks() -> list[Task]:
    tasks = []
    for kind, title in (
        ("gpqa", "GPQA Diamond"),
        ("gpqa-main", "GPQA Main"),
        ("gpqa-extended", "GPQA Extended"),
    ):
        task = Task(
            id=kind,
            title=title,
            group="science",
            protocol=(
                "OpenAI simple-evals 0-shot CoT. Choice order is shuffled with seed 0. One pass by default. "
                "--repeats 4 is the simple-evals average. --limit keeps the front of the list after repeats are expanded, "
                "so using it with --repeats keeps only the front of the first pass."
            ),
            blurb=_BLURBS[kind],
        )

        def build(opts, kind=kind):
            return _build(kind, opts)

        task.build = build
        task.grade_fn = grade_gpqa
        tasks.append(task)
    return tasks
