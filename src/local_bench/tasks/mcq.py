"""General-knowledge and commonsense multiple choice."""

from __future__ import annotations

from local_bench.tasks.base import Example, Task, shot_count
from local_bench.tasks.hfutil import (
    choice_pairs,
    gold_letter,
    grade_letter,
    grade_yes_no,
    load_hf,
    render_choices,
    zip_jsonl,
)

_LETTERS = "ABCDEFGHIJ"


def _header(subject: str) -> str:
    pretty = (subject or "the topic").replace("_", " ")
    return f"The following are multiple choice questions (with answers) about {pretty}."


def _mmlu_block(question: str, pairs: list[tuple[str, str]], gold: str | None) -> str:
    body = question.strip() + "\n\n" + render_choices(pairs)
    if gold is None:
        return body + "\nAnswer:"
    return body + f"\nAnswer: {gold}"


def build_mmlu(task: Task, opts) -> list[Example]:
    shots = shot_count(task, opts)
    test = list(load_hf("cais/mmlu", "all", "test"))
    dev_by_subject: dict[str, list[dict]] = {}
    if shots:
        for row in load_hf("cais/mmlu", "all", "dev"):
            dev_by_subject.setdefault(str(row["subject"]), []).append(row)
    examples = []
    for index, row in enumerate(test):
        subject = str(row["subject"])
        pairs = choice_pairs(row)
        gold = gold_letter(row, [label for label, _ in pairs])
        parts = [_header(subject), ""]
        for shot in dev_by_subject.get(subject, [])[:shots]:
            shot_pairs = choice_pairs(shot)
            shot_gold = gold_letter(shot, [label for label, _ in shot_pairs])
            parts.append(_mmlu_block(str(shot["question"]), shot_pairs, shot_gold))
            parts.append("")
        parts.append(_mmlu_block(str(row["question"]), pairs, None))
        examples.append(
            Example(
                id=f"{subject}-{index}",
                messages=[{"role": "user", "content": "\n".join(parts).strip()}],
                gold=gold,
                meta={"subject": subject, "allowed": _LETTERS[: len(pairs)], "category": subject},
            )
        )
    return examples


def _pro_block(question: str, pairs: list[tuple[str, str]], cot: str | None) -> str:
    lines = [f"Question: {question.strip()}", "Options:"]
    for label, text in pairs:
        lines.append(f"{label}. {text}")
    if cot:
        text = cot.replace("A: Let's think step by step.", "Answer: Let's think step by step.")
        lines.append(text.strip())
    else:
        lines.append("Answer: Let's think step by step.")
    return "\n".join(lines)


def _with_options(row) -> dict:
    data = dict(row)
    if data.get("options") is not None and not data.get("choices"):
        data["choices"] = list(data["options"])
    return data


def build_mmlu_pro(task: Task, opts) -> list[Example]:
    shots = shot_count(task, opts)
    test = list(load_hf("TIGER-Lab/MMLU-Pro", None, "test"))
    shots_by_cat: dict[str, list[dict]] = {}
    if shots:
        try:
            validation = list(load_hf("TIGER-Lab/MMLU-Pro", None, "validation"))
        except Exception:
            validation = []
        for row in validation:
            shots_by_cat.setdefault(str(row.get("category") or ""), []).append(row)
    examples = []
    for index, row in enumerate(test):
        category = str(row.get("category") or "")
        pairs = choice_pairs(_with_options(row))
        gold = str(row.get("answer") or gold_letter(row, [label for label, _ in pairs])).strip()
        intro = (
            f"The following are multiple choice questions (with answers) about {category}. "
            'Think step by step and then finish your answer with "the answer is (X)" '
            "where X is the correct letter choice."
        )
        parts = [intro, ""]
        for shot in shots_by_cat.get(category, [])[:shots]:
            shot_pairs = choice_pairs(_with_options(shot))
            parts.append(_pro_block(str(shot["question"]), shot_pairs, str(shot.get("cot_content") or "")))
            parts.append("")
        parts.append(_pro_block(str(row["question"]), pairs, None))
        examples.append(
            Example(
                id=str(row.get("question_id") or index),
                messages=[{"role": "user", "content": "\n".join(parts).strip()}],
                gold=gold.upper(),
                meta={"subject": category, "allowed": _LETTERS[: len(pairs)], "category": category},
            )
        )
    return examples


def build_kmmlu(task: Task, opts) -> list[Example]:
    del task, opts
    from datasets import get_dataset_config_names

    names = [name for name in get_dataset_config_names("HAERAE-HUB/KMMLU") if name not in {"default"}]
    if not names:
        names = ["default"]
    examples = []
    errors = []
    for name in names:
        try:
            rows = list(load_hf("HAERAE-HUB/KMMLU", name, "test"))
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        for index, row in enumerate(rows):
            subject = str(row.get("subject") or row.get("category") or name)
            pairs = choice_pairs(dict(row))
            raw_answer = row["answer"]
            if isinstance(raw_answer, int) or str(raw_answer).isdigit():
                # HAERAE stores 1 for A, 2 for B, 3 for C, 4 for D.
                gold = pairs[int(raw_answer) - 1][0]
            else:
                gold = str(raw_answer).strip()
            question = str(row.get("question") or row.get("query") or "")
            prompt = (
                f"다음은 {subject.replace('_', ' ')} 객관식 문제다. "
                "마지막 줄은 'Answer: X' 형식이고 X는 선택지 문자다.\n\n"
                + _mmlu_block(question, pairs, None)
            )
            examples.append(
                Example(
                    id=f"{subject}-{index}",
                    messages=[{"role": "user", "content": prompt}],
                    gold=str(gold).upper() if len(str(gold)) == 1 else str(gold),
                    meta={"subject": subject, "allowed": _LETTERS[: len(pairs)], "category": subject},
                )
            )
    if not examples:
        raise RuntimeError("KMMLU를 읽지 못했습니다. " + " | ".join(errors[:3]))
    return examples


def _arc(task: Task, opts, config: str) -> list[Example]:
    del task, opts
    rows = list(load_hf("allenai/ai2_arc", config, "test"))
    examples = []
    for index, row in enumerate(rows):
        pairs = choice_pairs(dict(row))
        gold = gold_letter(dict(row), [label for label, _ in pairs])
        prompt = (
            "Answer the science question. The last line must be 'Answer: X'.\n\n"
            + _mmlu_block(str(row["question"]), pairs, None)
        )
        examples.append(
            Example(
                id=str(row.get("id") or index),
                messages=[{"role": "user", "content": prompt}],
                gold=str(gold).upper() if len(str(gold)) == 1 else str(gold),
                meta={"subject": config, "allowed": "".join(label for label, _ in pairs), "category": config},
            )
        )
    return examples


def build_openbookqa(task: Task, opts) -> list[Example]:
    del task, opts
    rows = list(load_hf("allenai/openbookqa", "main", "test"))
    examples = []
    for index, row in enumerate(rows):
        pairs = choice_pairs(dict(row))
        gold = gold_letter(dict(row), [label for label, _ in pairs])
        stem = str(row.get("question_stem") or row.get("question"))
        prompt = "Answer the question. The last line must be 'Answer: X'.\n\n" + _mmlu_block(stem, pairs, None)
        examples.append(
            Example(
                id=str(row.get("id") or index),
                messages=[{"role": "user", "content": prompt}],
                gold=str(gold),
                meta={"allowed": "".join(label for label, _ in pairs), "subject": "openbookqa"},
            )
        )
    return examples


def build_commonsenseqa(task: Task, opts) -> list[Example]:
    del task, opts
    rows = list(load_hf("tau/commonsense_qa", None, "validation"))
    examples = []
    for index, row in enumerate(rows):
        pairs = choice_pairs(dict(row))
        gold = gold_letter(dict(row), [label for label, _ in pairs])
        prompt = (
            "Choose the commonsense answer. The last line must be 'Answer: X'.\n\n"
            + _mmlu_block(str(row["question"]), pairs, None)
        )
        examples.append(
            Example(
                id=str(row.get("id") or index),
                messages=[{"role": "user", "content": prompt}],
                gold=str(gold),
                meta={"allowed": "".join(label for label, _ in pairs), "subject": "commonsenseqa"},
            )
        )
    return examples


def build_boolq(task: Task, opts) -> list[Example]:
    del task, opts
    rows = list(load_hf("google/boolq", None, "validation"))
    examples = []
    for index, row in enumerate(rows):
        gold = "yes" if row["answer"] in (True, "true", "yes", 1) else "no"
        prompt = (
            f"Passage:\n{row['passage'].strip()}\n\n"
            f"Question: {row['question'].strip()}\n\n"
            "Answer yes or no. The last line must be 'Answer: yes' or 'Answer: no'."
        )
        examples.append(
            Example(
                id=str(index),
                messages=[{"role": "user", "content": prompt}],
                gold=gold,
                meta={"subject": "boolq"},
            )
        )
    return examples


def build_winogrande(task: Task, opts) -> list[Example]:
    del task, opts
    rows = list(load_hf("allenai/winogrande", "winogrande_xl", "validation"))
    examples = []
    for index, row in enumerate(rows):
        prompt = (
            "Choose the option that fills the blank (_).\n\n"
            f"{row['sentence']}\n\n"
            f"1. {row['option1']}\n"
            f"2. {row['option2']}\n\n"
            "The last line must be 'Answer: 1' or 'Answer: 2'."
        )
        examples.append(
            Example(
                id=str(index),
                messages=[{"role": "user", "content": prompt}],
                gold=str(row["answer"]).strip(),
                meta={"allowed": "12", "subject": "winogrande"},
            )
        )
    return examples


def build_truthfulqa(task: Task, opts) -> list[Example]:
    del task, opts
    rows = list(load_hf("truthfulqa/truthful_qa", "multiple_choice", "validation"))
    examples = []
    for index, row in enumerate(rows):
        targets = row["mc1_targets"]
        choices = list(targets["choices"])
        labels = list(targets["labels"])
        correct_index = next(pos for pos, label in enumerate(labels) if int(label) == 1)
        pairs = [(_LETTERS[pos], str(choice)) for pos, choice in enumerate(choices)]
        gold = pairs[correct_index][0]
        prompt = (
            "Pick the single truthful answer. The last line must be 'Answer: X'.\n\n"
            + _mmlu_block(str(row["question"]), pairs, None)
        )
        examples.append(
            Example(
                id=str(index),
                messages=[{"role": "user", "content": prompt}],
                gold=gold,
                meta={"allowed": "".join(label for label, _ in pairs), "subject": "truthfulqa"},
            )
        )
    return examples


def build_siqa(task: Task, opts) -> list[Example]:
    del task, opts
    rows = zip_jsonl(
        "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip",
        "socialiqa-train-dev.zip",
        "dev.jsonl",
        "dev-labels.lst",
    )
    examples = []
    for index, row in enumerate(rows):
        pairs = [("A", str(row["answerA"])), ("B", str(row["answerB"])), ("C", str(row["answerC"]))]
        gold = "ABC"[int(row["label"]) - 1]
        question = f"{row['context'].strip()}\n{row['question'].strip()}"
        prompt = (
            "Choose the socially plausible answer. The last line must be 'Answer: X'.\n\n"
            + _mmlu_block(question, pairs, None)
        )
        examples.append(
            Example(
                id=str(index),
                messages=[{"role": "user", "content": prompt}],
                gold=gold,
                meta={"allowed": "ABC", "subject": "siqa"},
            )
        )
    return examples


def build_hellaswag(task: Task, opts) -> list[Example]:
    del task, opts
    rows = list(load_hf("Rowan/hellaswag", None, "validation"))
    examples = []
    for index, row in enumerate(rows):
        context = str(row.get("ctx") or "").strip()
        if not context:
            context = f"{row.get('ctx_a', '')} {row.get('ctx_b', '')}".strip()
        activity = str(row.get("activity_label") or "").strip()
        question = f"{activity}: {context}" if activity else context
        endings = list(row["endings"])
        pairs = [(_LETTERS[pos], str(ending).strip()) for pos, ending in enumerate(endings)]
        gold = _LETTERS[int(row["label"])]
        prompt = (
            "Which ending is the most plausible continuation? "
            "The last line must be 'Answer: X'.\n\n"
            + _mmlu_block(question, pairs, None)
        )
        examples.append(
            Example(
                id=str(row.get("ind") or index),
                messages=[{"role": "user", "content": prompt}],
                gold=gold,
                meta={"allowed": "".join(label for label, _ in pairs), "subject": "hellaswag"},
            )
        )
    return examples


def build_piqa(task: Task, opts) -> list[Example]:
    del task, opts
    rows = zip_jsonl(
        "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip",
        "physicaliqa-train-dev.zip",
        "dev.jsonl",
        "dev-labels.lst",
    )
    examples = []
    for index, row in enumerate(rows):
        pairs = [("A", str(row["sol1"]).strip()), ("B", str(row["sol2"]).strip())]
        gold = "AB"[int(row["label"])]
        prompt = (
            "Which solution is more physically plausible? The last line must be 'Answer: X'.\n\n"
            + _mmlu_block(str(row["goal"]), pairs, None)
        )
        examples.append(
            Example(
                id=str(index),
                messages=[{"role": "user", "content": prompt}],
                gold=gold,
                meta={"allowed": "AB", "subject": "piqa"},
            )
        )
    return examples


def _bind(task: Task, builder) -> Task:
    task.build = lambda opts, task=task, builder=builder: builder(task, opts)
    task.grade_fn = grade_letter
    return task


def knowledge_tasks() -> list[Task]:
    mmlu = _bind(
        Task(
            id="mmlu",
            title="MMLU",
            group="knowledge",
            default_shots=5,
            protocol="생성형 5-shot. 원 논문의 로그확률 점수와는 다른 숫자다. 채팅 모델이 마지막 Answer 문자로 답한다.",
            blurb="57개 과목 일반지식. 약 1만 4천 문항.",
        ),
        build_mmlu,
    )
    mmlu_pro = _bind(
        Task(
            id="mmlu-pro",
            title="MMLU-Pro",
            group="knowledge",
            default_shots=0,
            protocol="TIGER-Lab 0-shot CoT. 논문의 5-shot CoT는 --shots 5. 10지선다에 가깝다.",
            blurb="MMLU보다 어려운 일반지식. 보기가 최대 10개.",
        ),
        build_mmlu_pro,
    )
    kmmlu = _bind(
        Task(
            id="kmmlu",
            title="KMMLU",
            group="knowledge",
            protocol="HAERAE KMMLU test. 한국어 객관식, 마지막 Answer 문자.",
            blurb="한국어 전공지식 객관식 약 3만 5천 문항. --subject Accounting 처럼 과목만 돌릴 수 있다.",
        ),
        build_kmmlu,
    )
    arc = _bind(
        Task(
            id="arc-challenge",
            title="ARC-Challenge",
            group="knowledge",
            protocol="AI2 ARC-Challenge test, 0-shot 생성형.",
            blurb="초등 과학이지만 검색 없이 어려운 문항 1172개.",
        ),
        lambda task, opts: _arc(task, opts, "ARC-Challenge"),
    )
    arc_easy = _bind(
        Task(
            id="arc-easy",
            title="ARC-Easy",
            group="knowledge",
            protocol="AI2 ARC-Easy test, 0-shot 생성형.",
            blurb="ARC의 쉬운 과학 문항.",
        ),
        lambda task, opts: _arc(task, opts, "ARC-Easy"),
    )
    openbook = _bind(
        Task(
            id="openbookqa",
            title="OpenBookQA",
            group="knowledge",
            protocol="OpenBookQA test, 0-shot 생성형.",
            blurb="초등 과학 상식 500문항. 사실 하나를 알고 적용해야 한다.",
        ),
        build_openbookqa,
    )
    csqa = _bind(
        Task(
            id="commonsenseqa",
            title="CommonsenseQA",
            group="knowledge",
            protocol="CommonsenseQA validation. 테스트 셋 정답은 비공개라 validation을 쓴다.",
            blurb="일상 상식 객관식.",
        ),
        build_commonsenseqa,
    )
    boolq = Task(
        id="boolq",
        title="BoolQ",
        group="knowledge",
        protocol="BoolQ validation. yes/no. 테스트 정답은 비공개.",
        blurb="짧은 글을 읽고 예/아니오로 답하는 질문.",
        build=lambda opts: build_boolq(None, opts),
        grade_fn=grade_yes_no,
    )
    wino = _bind(
        Task(
            id="winogrande",
            title="WinoGrande",
            group="knowledge",
            protocol="WinoGrande-XL validation. 테스트 정답은 비공개.",
            blurb="문장 빈칸에 들어갈 대상을 고르는 상식.",
        ),
        build_winogrande,
    )
    truthful = _bind(
        Task(
            id="truthfulqa",
            title="TruthfulQA",
            group="knowledge",
            protocol="TruthfulQA MC1 validation. 정답이 하나인 보기만 맞으면 된다.",
            blurb="사람들이 자주 틀리는 사실 질문. 진실한 보기 하나를 고른다.",
        ),
        build_truthfulqa,
    )
    siqa = _bind(
        Task(
            id="siqa",
            title="SocialIQA",
            group="knowledge",
            protocol="SocialIQA validation.",
            blurb="상황 속에서 사람의 의도나 반응을 고르는 사회 상식.",
        ),
        build_siqa,
    )
    hellaswag = _bind(
        Task(
            id="hellaswag",
            title="HellaSwag",
            group="knowledge",
            protocol="생성형 validation. 공식 리더보드는 로그확률이라 이 점수는 그 숫자와 다르다.",
            blurb="문장 뒤에 이어질 자연스러운 끝을 고른다.",
        ),
        build_hellaswag,
    )
    piqa = _bind(
        Task(
            id="piqa",
            title="PIQA",
            group="knowledge",
            protocol="생성형 validation. 공식 점수는 로그확률이다.",
            blurb="물리적으로 말이 되는 해결책을 둘 중 고른다.",
        ),
        build_piqa,
    )
    return [mmlu, mmlu_pro, kmmlu, arc, arc_easy, openbook, csqa, boolq, wino, truthful, siqa, hellaswag, piqa]
