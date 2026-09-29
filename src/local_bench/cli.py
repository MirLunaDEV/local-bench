"""Command line for local benchmark runs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from local_bench.catalog import SUITES, all_tasks, resolve_targets
from local_bench.client import ChatClient
from local_bench.discover import discover, resolve_endpoint
from local_bench.runner import format_summary, load_summaries, run_task


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def load_local_env(path: Path | None = None) -> None:
    """Fill missing environment variables from a local .env file.

    Existing process variables stay as they are. Blank values and comments are
    skipped. Values are never logged.
    """
    env_path = path if path is not None else Path.cwd() / ".env"
    try:
        if not env_path.is_file():
            return
        text = env_path.read_text(encoding="utf-8-sig")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        if not (key[0].isalpha() or key[0] == "_") or not all(ch.isalnum() or ch == "_" for ch in key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value == "":
            continue
        os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-bench",
        description="Run public LLM benchmarks on a local OpenAI-compatible model.",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or "local")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", help="켜져 있는 로컬 서버와 모델 id를 보여 줍니다")

    listing = sub.add_parser("list", help="벤치와 스위트 목록을 보여 줍니다")
    listing.add_argument("--suite", help="이 스위트에 들어 있는 태스크만 표시합니다")

    run = sub.add_parser("run", help="벤치 또는 스위트를 실행합니다")
    run.add_argument("target", help="태스크 id, 스위트 이름, 또는 쉼표로 이은 목록입니다")
    run.add_argument("--model", help="서버가 보여주는 모델 id입니다. 생략하면 목록에서 고릅니다")
    run.add_argument("--base-url", help="OpenAI 호환 주소입니다. 예: http://127.0.0.1:11434/v1")
    run.add_argument("--limit", type=int, help="앞에서 N개만 채점합니다. 전체 점수와는 비교하지 않습니다")
    run.add_argument("--shots", type=int, help="few-shot 개수입니다. 생략하면 벤치 기본값을 사용합니다")
    run.add_argument("--repeats", type=int, default=1, help="GPQA 반복 횟수입니다. simple-evals는 4입니다")
    run.add_argument("--subject", help="mmlu, kmmlu, mmlu-pro 과목 필터입니다")
    run.add_argument("--category", help="bfcl 범주입니다. 예: simple_python,multiple 또는 all")
    run.add_argument("--lang", help="mgsm 언어입니다. 예: en, ko는 없습니다, zh, ja")
    run.add_argument("--workers", type=int, default=1, help="동시 요청 수입니다. 로컬 GPU는 1이 안전합니다")
    run.add_argument("--max-tokens", type=int, default=4096)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--timeout", type=float, default=300, help="요청 하나당 제한 시간(초)입니다")
    run.add_argument("--max-errors", type=int, default=8, help="연속 호출 실패가 이 횟수면 멈춥니다")
    run.add_argument("--no-think", action="store_true", help="서버가 지원하면 enable_thinking=false로 둡니다")
    run.add_argument("--fresh", action="store_true", help="저장된 답을 무시하고 다시 받습니다")
    run.add_argument("--baseline-model", help="arena-hard 비교 기준 모델입니다")
    run.add_argument("--baseline-url", help="기준 모델 서버입니다. 생략하면 같은 서버를 사용합니다")
    run.add_argument("--judge-model", help="arena-hard 심판 모델입니다")
    run.add_argument("--judge-url", help="심판 모델 서버입니다. 생략하면 같은 서버를 사용합니다")
    run.add_argument("--runs-dir", default="runs", help="결과 디렉터리입니다")

    report = sub.add_parser("report", help="저장된 점수 요약을 보여 줍니다")
    report.add_argument("--model", help="모델 id에 이 문자열이 들어간 결과만 보여 줍니다")
    report.add_argument("--runs-dir", default="runs")
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    load_local_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "models":
        return cmd_models(args.api_key)
    if args.cmd == "list":
        return cmd_list(args.suite)
    if args.cmd == "report":
        return cmd_report(Path(args.runs_dir), args.model)
    if args.cmd == "run":
        return cmd_run(args)
    parser.error(f"알 수 없는 명령 {args.cmd}")
    return 2


def cmd_models(api_key: str) -> int:
    rows = discover(api_key=api_key)
    if not rows:
        print("실행 중인 로컬 서버가 없습니다.")
        print("Ollama 11434, LM Studio 1234, llama.cpp 8080, vLLM 8000, Q38 1923 을 확인하세요.")
        return 1
    for row in rows:
        print(f"{row['model']}\t{row['backend']}\t{row['base_url']}")
    return 0


def cmd_list(suite: str | None) -> int:
    tasks = all_tasks()
    if suite:
        from local_bench.catalog import suite_ids

        wanted = set(suite_ids(suite))
        tasks = [task for task in tasks if task.id in wanted]
        print(f"스위트 {suite}")
    current = None
    for task in tasks:
        if task.group != current:
            current = task.group
            print(f"\n[{current}]")
        print(f"  {task.id:<16} {task.title}")
        if task.blurb:
            print(f"    {task.blurb}")
    if suite:
        return 0
    print("\n[스위트]")
    print("  all              arena-hard를 뺀 전체입니다")
    for name, ids in SUITES.items():
        print(f"  {name:<16} {', '.join(ids)}")
    return 0


def cmd_report(runs_dir: Path, model: str | None) -> int:
    rows = load_summaries(runs_dir, model)
    if not rows:
        print(f"{runs_dir} 에 요약이 없습니다.")
        return 1
    for row in rows:
        print(format_summary(row))
        print()
    return 0


def cmd_run(args) -> int:
    targets = resolve_targets(args.target)
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    base_url, model = resolve_endpoint(args.model, args.base_url, args.api_key, interactive)
    client = ChatClient(
        base_url=base_url,
        model=model,
        api_key=args.api_key,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        no_think=args.no_think,
    )
    print(f"모델 {model}  @ {client.base_url}", file=sys.stderr)
    code_ids = {"humaneval", "mbpp"}
    if any(task.id in code_ids for task in targets):
        print(
            "HumanEval/MBPP는 모델이 만든 파이썬을 이 PC에서 실행합니다. "
            "시간 제한이 있고 os.kill, os.system 같은 함수 몇 개를 비우지만, 이것은 보안 샌드박스가 아닙니다. "
            "importlib.reload(os)로 그 차단은 바로 되돌려지고, 파일 읽기, 네트워크, os.startfile, os.spawn* 은 열려 있습니다. "
            "믿는 모델만, 지워도 되는 환경에서 돌리세요.",
            file=sys.stderr,
        )
    runs_dir = Path(args.runs_dir)
    failures = []
    for task in targets:
        try:
            summary = run_task(task, client, args, runs_dir)
        except KeyboardInterrupt:
            print("\n사용자가 중단했습니다.", file=sys.stderr)
            return 130
        except Exception as exc:
            failures.append((task.id, str(exc)))
            print(f"실패 {task.id}: {exc}", file=sys.stderr)
            continue
        print(format_summary(summary))
        print()
    if failures:
        print("끝나지 않은 벤치: " + ", ".join(task_id for task_id, _ in failures), file=sys.stderr)
        return 1
    return 0
