# local-bench

[한국어](README.ko.md)

Pick a local model and run the **public** benchmarks that Artificial Analysis and LM Arena use. The server only needs an OpenAI-compatible `POST /v1/chat/completions`. Ollama, LM Studio, llama.cpp, vLLM, and Q38 in this folder (port 1923) speak that protocol.

Each score follows that benchmark's public grading rule. A run with `--limit` scores only the front of the set. Treat that number as a slice, and keep it separate from a full leaderboard score.

## Install

```powershell
cd local-bench
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\local-bench list
```

The commands below assume this virtual environment is active. Otherwise call `.\.venv\Scripts\local-bench`.

```powershell
.\.venv\Scripts\Activate.ps1
```

The first run downloads question files from Hugging Face and the official repositories into `~/.cache/local-bench` and the Hugging Face cache. Question files stay out of this repository. Only the BBH 3-shot exemplars are included, as grading prompts. The GPQA authors ask that the questions not be posted on the web as-is.

Keys stay out of the repository. A local server works without an API key. Set a remote server or the cache directory with shell environment variables. When the folder where you start the program contains a `.env` file, the program fills only names that are not already set in the shell. Empty values are skipped. `.env.example` lists the names, and `.env` is gitignored.

| Variable | Role |
|---|---|
| `OPENAI_API_KEY` | Key for a remote server. Empty means `local` |
| `LOCAL_BENCH_BASE_URL` | Extra OpenAI-compatible URL to probe |
| `LOCAL_BENCH_CACHE` | Question cache directory. Default is `~/.cache/local-bench` |

## Choose a model

```powershell
local-bench models
local-bench list
local-bench run gpqa
```

If `run` has no `--model`, the program asks for a model number from the servers already running in the terminal.

```powershell
local-bench run gpqa --base-url http://127.0.0.1:1923/v1 --model Qwen3.8-Flash-Next-NVFP4 --limit 5
local-bench run knowledge --model llama3.1:8b --limit 20
local-bench report
```

Results accumulate in `runs/<model>/<bench>.jsonl` and `.summary.json`. The same command again skips questions that already have answers. `--fresh` requests them from the start.

A local GPU often serves one request at a time. The default is `--workers 1`. Raise `--max-tokens` for reasoning models. When the server supports it, `--no-think` turns thinking tokens off.

## What it runs

| Group | Benchmarks | Scoring |
|---|---|---|
| Science | `gpqa`, `gpqa-main`, `gpqa-extended` | simple-evals 0-shot CoT. Choices are shuffled with seed 0. `--repeats 4` is the simple-evals average |
| General knowledge | `mmlu`, `mmlu-pro`, `kmmlu` | MMLU is generative 5-shot. MMLU-Pro is 0-shot CoT; the paper setting is `--shots 5`. KMMLU is Korean |
| Science and commonsense | `arc-challenge`, `arc-easy`, `openbookqa`, `commonsenseqa`, `boolq`, `winogrande`, `truthfulqa`, `siqa` | Last `Answer:` line. Sets with private test labels use validation |
| Generative commonsense | `hellaswag`, `piqa` | The model picks a choice. The official table uses log probabilities, which is a different protocol |
| Math | `gsm8k`, `math500`, `aime2024`, `mgsm` | GSM8K is 8-shot CoT. MATH-500 and AIME use `\boxed{}`. MATH-500 compares strings and floats only. MGSM takes `--lang en` |
| Code | `humaneval`, `mbpp` | pass@1. The Python the model writes is executed on this machine |
| Instruction | `ifeval` | The headline score is prompt-level strict. Instruction-level strict and loose are reported too |
| Reasoning and reading | `bbh`, `musr`, `drop` | BBH is exact match after Suzgun 3-shot CoT. MuSR is multiple choice. DROP is token F1 |
| Agent | `bfcl` | AST accuracy on the BFCL v4 Python prompting track |
| Arena | `arena-hard` | Arena-Hard-Auto v2. It needs a baseline model and a judge model |

Suites:

- `quick` — GPQA Diamond, HumanEval, AIME 2024, OpenBookQA, IFEval
- `core` — the short list above, plus full MMLU, GSM8K, ARC-Challenge, and BFCL
- `knowledge`, `science`, `math`, `code`, `instruction`, `reasoning`, `reading`, `agent`, `arena`
- `aa-public` — public tasks Artificial Analysis has used: GPQA, MMLU-Pro, GSM8K, MATH-500, AIME, HumanEval, IFEval, BFCL
- `all` — everything except `arena-hard`

```powershell
local-bench run bfcl --category simple_python --limit 20
local-bench run mgsm --lang en --limit 20
local-bench run mmlu --subject abstract_algebra
local-bench run arena-hard --model my-model --baseline-model baseline --judge-model judge
```

## Where the number matches a leaderboard

Much of the weight in Artificial Analysis Intelligence Index v4.3 sits on private sets. AA-Briefcase, AutomationBench-AA, AA-Omniscience, and CritPt have no public questions, so this runner does not reproduce them. Terminal-Bench and SciCode are public, and they need Docker plus a scientific code sandbox, which is outside this runner. GPQA Diamond left the September 2026 index and remains a separately published science score for new models.

LM Arena Elo comes from human votes. An offline run cannot produce that Elo. `arena-hard` approximates the ranking with questions and a judge prompt published by LM Arena. The public table is tied to o3-mini baseline answers and a GPT-4.1 judge. A score of 0.5 here means a tie with the baseline model you chose.

The original MMLU paper, and the official HellaSwag and PIQA scores, use next-token log probabilities. Many local chat servers do not return those probabilities, so this runner grades the letter the model chooses.

The BFCL score is the Python prompting track. The default categories are simple, multiple, parallel, and parallel_multiple, and the score is the average of the category scores. `--category all` includes live and irrelevance. Multi-turn state environments, Java, and JavaScript belong to the official BFCL repository.

HumanEval and MBPP run generated code in a subprocess. There is a time limit, and a few functions such as `os.kill` and `os.system` are cleared. This is not a security sandbox. `importlib.reload(os)` restores that block immediately. File reads, network access, `os.startfile`, and `os.spawn*` stay available. Run it on a model you trust, in an environment you can wipe.

IFEval's sentence count uses the regex split in the same file, in place of NLTK punkt. Only `number_sentences` items can differ slightly from the official script.

The BBH default is the per-task 3-shot CoT prompt published by Suzgun et al. The exemplars are prepended, the prompt ends with `A: Let's think step by step.`, and the grader normalizes the text after the last `the answer is` or `Answer:`. Multiple choice treats `(B)` and `B` as the same answer. `--shots 0` attaches no exemplars, so that run is separate from the paper's BBH number. The 3-object, 5-object, and 7-object variants of logical deduction and tracking shuffled objects share one published prompt file. This runner uses that file's 3-object exemplars as published.

MATH-500 does not check symbolic equivalence. Two answers match when the strings are equal after whitespace and notation such as `\dfrac` are normalized, or when both sides parse as floats. `\frac{1}{2}` and `0.5` are different answers. This score can land below an official MATH grader that uses sympy.

In GPQA, `--limit` keeps the first N items of the list after repeats are expanded. `--repeats 4 --limit 20` is the first 20 items of the first pass, which is a different number from a four-pass average.

When a response includes `usage`, the result row stores the token counts and the summary prints generated tok/s. If the server omits usage, that field stays empty. On HTTP 429 or 503, a `Retry-After` header sets the wait before the next try. The cap is 60 seconds.

The original code is under the MIT license. The IFEval checker, the BFCL checker, and the BBH CoT prompts are Apache-2.0. Sources are listed in `src/local_bench/third_party/NOTICE`.
