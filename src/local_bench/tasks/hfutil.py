"""Small helpers for Hugging Face multiple-choice rows."""

from __future__ import annotations

import io
import json
import zipfile

from local_bench.download import cached_download
from local_bench.scoreutil import extract_letter, extract_yes_no
from local_bench.tasks.base import Example


def zip_jsonl(url: str, cache_name: str, jsonl_suffix: str, label_suffix: str) -> list[dict]:
    """Read a JSONL plus a line-aligned label file from a zip the benchmark publishes."""
    path = cached_download(url, cache_name)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()

        def find(suffix: str) -> str:
            matches = []
            for name in names:
                normal = name.replace("\\", "/")
                leaf = normal.rsplit("/", 1)[-1]
                if "__MACOSX" in normal or leaf.startswith("._"):
                    continue
                if normal.endswith(suffix):
                    matches.append(name)
            if len(matches) != 1:
                raise FileNotFoundError(f"{suffix} not found uniquely in {cache_name}: {matches}")
            return matches[0]

        with archive.open(find(jsonl_suffix)) as handle:
            rows = [json.loads(line) for line in io.TextIOWrapper(handle, encoding="utf-8") if line.strip()]
        with archive.open(find(label_suffix)) as handle:
            labels = [line.strip() for line in io.TextIOWrapper(handle, encoding="utf-8") if line.strip()]
    if len(rows) != len(labels):
        raise RuntimeError(f"{cache_name}: {len(rows)} rows and {len(labels)} labels")
    for row, label in zip(rows, labels):
        row["label"] = label
    return rows


def load_hf(path: str, config: str | None, split: str):
    from datasets import load_dataset

    if config is None:
        return load_dataset(path, split=split)
    return load_dataset(path, config, split=split)


def choice_pairs(row: dict, letters: str = "ABCDEFGHIJ") -> list[tuple[str, str]]:
    choices = row.get("choices")
    if isinstance(choices, dict) and "text" in choices:
        labels = list(choices.get("label") or letters[: len(choices["text"])])
        texts = list(choices["text"])
        return [(str(label), str(text)) for label, text in zip(labels, texts)]
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return [(str(item.get("label")), str(item.get("text"))) for item in choices]
    if isinstance(choices, list):
        return [(letters[index], str(text)) for index, text in enumerate(choices)]
    paired = []
    for label in letters:
        if label in row and row[label] not in (None, ""):
            paired.append((label, str(row[label])))
    if paired:
        return paired
    raise ValueError("선택지를 읽지 못했습니다.")


def gold_letter(row: dict, labels: list[str]) -> str:
    for key in ("answerKey", "answer", "label", "target"):
        if key not in row or row[key] in (None, ""):
            continue
        value = row[key]
        label_set = {str(label): str(label) for label in labels}
        upper = {str(label).upper(): str(label) for label in labels}
        if isinstance(value, int) and 0 <= value < len(labels):
            return str(labels[value])
        text = str(value).strip()
        if text in label_set:
            return label_set[text]
        if text.upper() in upper:
            return upper[text.upper()]
        if text.isdigit():
            index = int(text)
            if 0 <= index < len(labels):
                return str(labels[index])
    raise ValueError(f"정답 칸을 읽지 못했습니다: {sorted(row.keys())}")


def render_choices(pairs: list[tuple[str, str]], style: str = "dot") -> str:
    lines = []
    for label, text in pairs:
        if style == "paren":
            lines.append(f"{label}) {text}")
        else:
            lines.append(f"{label}. {text}")
    return "\n".join(lines)


def user_example(item_id: str, prompt: str, gold: str, subject: str | None = None, allowed: str | None = None) -> Example:
    labels = allowed or "".join(dict.fromkeys(gold))
    return Example(
        id=str(item_id),
        messages=[{"role": "user", "content": prompt}],
        gold=gold,
        meta={"subject": subject, "allowed": allowed or "ABCDEFGHIJ", "category": subject},
    )


def grade_letter(example: Example, text: str) -> dict:
    allowed = example.meta.get("allowed") or "ABCD"
    pred = extract_letter(text, allowed)
    gold = str(example.gold).strip()
    if len(gold) == 1:
        gold = gold.upper()
    return {"score": 1.0 if pred == gold else 0.0, "pred": pred}


def grade_yes_no(example: Example, text: str) -> dict:
    pred = extract_yes_no(text)
    gold = str(example.gold).strip().lower()
    return {"score": 1.0 if pred == gold else 0.0, "pred": pred}
