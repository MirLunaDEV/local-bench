"""Answer extraction shared by the multiple-choice and short-answer benches."""

from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_PIPE = re.compile(r"<\|think\|>.*?<\|/think\|>", re.DOTALL)


def visible_text(text: str) -> str:
    """Drop reasoning traces so the grader reads the answer the user would see."""
    cleaned = _THINK_BLOCK.sub("", text or "")
    cleaned = _THINK_PIPE.sub("", cleaned)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    return cleaned.strip()


def _letter_commitment(text: str, match: re.Match[str]) -> bool:
    """Reject the article in 'the answer is a good ...'."""
    letter = match.group(1)
    after = text[match.end(1) :]
    if after[:1].isalpha():
        return False
    if letter.islower() and re.match(r"[.!]?\s+[A-Za-z]", after):
        return False
    return True


def extract_letter(text: str, allowed: str = "ABCD") -> str | None:
    """Return the last explicit choice letter.

    Local reasoning models often mention a letter while thinking and then
    commit on a final `Answer:` line. The last commitment is the one scored.
    A lowercase "a" followed by more words is an article, not a choice.
    """
    if not text:
        return None
    allowed_set = {ch.upper() for ch in allowed}
    class_ = "".join(re.escape(ch) for ch in sorted(allowed_set))
    pattern = re.compile(
        rf"(?i)(?:the\s+answer\s+is|answer\s+is|answer|정답|답변)\s*[:：]?\s*\$?\(?([{class_}])\)?"
    )
    chosen = None
    for match in pattern.finditer(text):
        if not _letter_commitment(text, match):
            continue
        letter = match.group(1).upper()
        if letter in allowed_set:
            chosen = letter
    if chosen:
        return chosen
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    last = lines[-1].strip("`*_ ").rstrip(".").strip()
    boxed = re.fullmatch(rf"(?i)\(?([{class_}])\)?", last)
    if boxed and boxed.group(1).upper() in allowed_set:
        return boxed.group(1).upper()
    return None


def extract_yes_no(text: str) -> str | None:
    if not text:
        return None
    labeled = list(re.finditer(r"(?i)(?:answer|정답|답변)\s*[:：]\s*(yes|no|true|false)\b", text))
    if labeled:
        token = labeled[-1].group(1).lower()
        return "yes" if token in {"yes", "true"} else "no"
    loose = list(re.finditer(r"(?i)\b(yes|no)\b", text))
    if not loose:
        return None
    return loose[-1].group(1).lower()


def extract_last_number(text: str) -> str | None:
    if not text:
        return None
    chunk = text.split("####")[-1] if "####" in text else text
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", chunk)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def numbers_equal(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(str(gold).replace(",", ""))) <= 1e-6
    except ValueError:
        return False


def extract_boxed(text: str) -> str | None:
    if not text:
        return None
    key = r"\boxed"
    start = text.rfind(key)
    if start < 0:
        return None
    index = start + len(key)
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != "{":
        return None
    depth = 0
    chars: list[str] = []
    for char in text[index:]:
        if char == "{":
            depth += 1
            if depth == 1:
                continue
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        if depth >= 1:
            chars.append(char)
    return None


def normalize_math(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = text.replace("$", "")
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    text = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace(r"\,", "").replace(r"\!", "").replace(r"\;", "")
    text = text.replace(" ", "").replace("\n", "")
    text = text.rstrip(".").lower()
    return text


def math_equal(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    left = normalize_math(pred)
    right = normalize_math(gold)
    if left == right:
        return True
    return numbers_equal(left, right)


def normalize_tokens(text: str) -> list[str]:
    lowered = text.lower()
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return [tok for tok in lowered.split() if tok]


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_tokens(prediction)
    gold_tokens = normalize_tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in gold_tokens:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        if counts.get(token, 0) > 0:
            overlap += 1
            counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def short_answer(text: str) -> str:
    """Prefer the last explicit commitment over the whole reasoning trace.

    Official BBH exemplars end with ``So the answer is X.`` and have no colon.
    ``Answer: X`` is the form this runner asks for on other short-answer tasks.
    """
    if not text:
        return ""
    labeled = list(
        re.finditer(
            r"(?im)(?:the\s+answer\s+is\s*[:：]?|(?:final answer|answer)\s*[:：])\s*([^\n]+)",
            text,
        )
    )
    if labeled:
        return labeled[-1].group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()
