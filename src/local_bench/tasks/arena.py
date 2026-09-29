"""Arena-Hard-Auto v2. Local models supply the answer, the baseline, and the judge."""

from __future__ import annotations

import json
import sys

from local_bench.client import ChatClient
from local_bench.download import cached_download, load_json_records
from local_bench.runner import JsonlStore, slug
from local_bench.scoreutil import visible_text
from local_bench.tasks.base import mean_metrics

_QUESTIONS = "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/data/arena-hard-v2.0/question.jsonl"

_JUDGE = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two "
    "AI assistants to the user prompt displayed below. You will be given assistant A's answer and "
    "assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\n"
    "Begin your evaluation by generating your own answer to the prompt. You must provide your "
    "answers before judging any answers.\n\n"
    "When evaluating the assistants' answers, compare both assistants' answers with your answer. "
    "You must identify and correct any mistakes or inaccurate information.\n\n"
    "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means "
    "the answer correctly responds to the prompt or follows the instructions. Note when user prompt "
    "has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask "
    "for clarifications or more information from the user than providing an answer based on assumptions. "
    "Relevant means all parts of the response closely connect or are appropriate to what is being asked. "
    "Concise means the response is clear and not verbose or excessive.\n\n"
    "Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify "
    "any missing important information in the assistants' answers that would be beneficial to include "
    "when responding to the user prompt.\n\n"
    "After providing your explanation, you must output only one of the following choices as your final "
    "verdict with a label:\n\n"
    "1. Assistant A is significantly better: [[A>>B]]\n"
    "2. Assistant A is slightly better: [[A>B]]\n"
    "3. Tie, relatively the same: [[A=B]]\n"
    "4. Assistant B is slightly better: [[B>A]]\n"
    "5. Assistant B is significantly better: [[B>>A]]\n\n"
    'Example output: "My final verdict is tie: [[A=B]]".'
)

_CREATIVE = _JUDGE.replace(
    "Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\n",
    "",
)

_WEIGHTS = {
    "A>>B": {"A": 1.0, "B": 0.0},
    "A>B": {"A": 0.75, "B": 0.25},
    "A=B": {"A": 0.5, "B": 0.5},
    "B>A": {"A": 0.25, "B": 0.75},
    "B>>A": {"A": 0.0, "B": 1.0},
}

_USER = """<|User Prompt|>
{question}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>"""


def _load_questions() -> list[dict]:
    path = cached_download(_QUESTIONS, "arena-hard-v2.0-question.jsonl")
    return load_json_records(path.read_text(encoding="utf-8"))


def _verdict(text: str) -> str | None:
    import re

    found = re.findall(r"\[\[(A>>B|A>B|A=B|B>A|B>>A)\]\]", text or "")
    return found[-1] if found else None


def _judge_system(row: dict) -> str:
    category = str(row.get("subcategory") or row.get("category") or "")
    if category == "creative_writing":
        return _CREATIVE
    return _JUDGE


class ArenaTask:
    id = "arena-hard"
    title = "Arena-Hard v2"
    group = "arena"
    metric = "score"
    default_shots = 0
    blurb = "LM Arena의 자동 평가. 기준 모델과 심판 모델이 더 필요하다. 공개 Elo 그 자체는 아니다."
    protocol = (
        "Arena-Hard-Auto v2 질문 750개. 기준 답과 맞대어 두 번(자리 바꿈) 심판한다. "
        "점수 0.5는 기준 모델과 비등. 공개 리더보드 숫자는 o3-mini 기준 답과 GPT-4.1 심판에 묶여 있으므로, "
        "로컬 심판 점수는 그 표와 직접 비교되지 않는다."
    )

    def execute(self, client, opts, runs_dir):
        if not opts.baseline_model or not opts.judge_model:
            raise RuntimeError(
                "arena-hard 는 --baseline-model 과 --judge-model 이 필요합니다. "
                "기준 모델은 비교 대상, 심판 모델은 둘 중 누가 나은지 고릅니다."
            )
        questions = _load_questions()
        if opts.limit:
            questions = questions[: int(opts.limit)]
            print(
                f"arena-hard: 앞에서 {len(questions)}개만 채점합니다. 리더보드와 비교하려면 --limit 없이 실행하세요.",
                file=sys.stderr,
            )
        baseline = _client_like(client, opts.baseline_model, opts.baseline_url or client.base_url, opts)
        judge = _client_like(client, opts.judge_model, opts.judge_url or client.base_url, opts)
        base_store = JsonlStore(runs_dir / slug(baseline.model) / "arena-hard-v2.baseline.jsonl")
        path = runs_dir / slug(client.model) / "arena-hard.jsonl"
        if getattr(opts, "fresh", False) and path.exists():
            path.unlink()
        store = JsonlStore(path)
        for index, row in enumerate(questions, start=1):
            uid = str(row.get("uid") or index)
            existing = store.get(uid)
            if existing and not existing.get("error") and existing.get("metrics"):
                print(f"[{index}/{len(questions)}] arena-hard {uid} 기존 결과", file=sys.stderr)
                continue
            try:
                answer = (existing or {}).get("answer") or client.chat(
                    [{"role": "user", "content": row["prompt"]}]
                )
                baseline_answer = _baseline_answer(base_store, baseline, uid, row)
                score, verdicts = _judge_pair(judge, row, answer, baseline_answer)
                record = {
                    "id": uid,
                    "category": row.get("category"),
                    "gold": baseline.model,
                    "response": answer[:100000],
                    "answer": answer[:100000],
                    "baseline_answer": baseline_answer[:20000],
                    "verdicts": verdicts,
                    "metrics": {"score": score},
                    "error": None,
                }
            except Exception as exc:
                record = {
                    "id": uid,
                    "category": row.get("category"),
                    "gold": baseline.model,
                    "response": "",
                    "metrics": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
            store.add(record)
            shown = record["metrics"].get("score") if not record["error"] else record["error"][:120]
            print(f"[{index}/{len(questions)}] arena-hard {uid} {shown}", file=sys.stderr)
        rows = list(store.rows.values())
        metrics = mean_metrics(rows)
        scored = [row for row in rows if not row.get("error") and isinstance((row.get("metrics") or {}).get("score"), (int, float))]
        if scored:
            wins = sum(1 for row in scored if row["metrics"]["score"] > 0.5)
            ties = sum(1 for row in scored if row["metrics"]["score"] == 0.5)
            metrics["win_rate"] = wins / len(scored)
            metrics["tie_rate"] = ties / len(scored)
            metrics["loss_rate"] = 1 - metrics["win_rate"] - metrics["tie_rate"]
        summary = {
            "task": self.id,
            "title": self.title,
            "model": client.model,
            "base_url": client.base_url,
            "protocol": self.protocol,
            "primary_metric": "score",
            "metrics": metrics,
            "limit": opts.limit,
            "baseline_model": baseline.model,
            "judge_model": judge.model,
            "path": str(path),
        }
        path.with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary


def _client_like(client, model: str, base_url: str, opts) -> ChatClient:
    if model == client.model and base_url.rstrip("/") == client.base_url.rstrip("/"):
        return client
    return ChatClient(
        base_url=base_url,
        model=model,
        api_key=getattr(client, "api_key", "local"),
        timeout=client.timeout,
        temperature=client.temperature,
        max_tokens=client.max_tokens,
        no_think=client.no_think,
    )


def _baseline_answer(store: JsonlStore, client: ChatClient, uid: str, row: dict) -> str:
    existing = store.get(uid)
    if existing and existing.get("answer"):
        return existing["answer"]
    answer = client.chat([{"role": "user", "content": row["prompt"]}])
    store.add({"id": uid, "answer": answer, "error": None})
    return answer


def _judge_pair(judge: ChatClient, row: dict, answer: str, baseline_answer: str) -> tuple[float, list[str]]:
    system = _judge_system(row)
    limit = max(judge.max_tokens, 2048)
    scores = []
    verdicts = []
    # Game 1: baseline is A, candidate is B. Game 2 swaps them.
    seats = (("B", baseline_answer, answer), ("A", answer, baseline_answer))
    for seat, answer_a, answer_b in seats:
        raw = judge.chat(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": _USER.format(
                        question=row["prompt"],
                        answer_a=visible_text(answer_a),
                        answer_b=visible_text(answer_b),
                    ),
                },
            ],
            max_tokens=limit,
        )
        verdict = _verdict(visible_text(raw))
        if verdict is None:
            raise RuntimeError("심판 응답에서 [[A>>B]] 형식의 판정을 찾지 못했습니다.")
        verdicts.append(verdict)
        scores.append(_WEIGHTS[verdict][seat])
    return sum(scores) / len(scores), verdicts


def arena_task() -> ArenaTask:
    return ArenaTask()
