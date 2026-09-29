# local-bench

[English](README.md)

로컬 모델을 골라, Artificial Analysis와 LM Arena가 사용하는 **공개** 벤치마크를 실행합니다. 서버는 OpenAI 호환 `POST /v1/chat/completions`만 있으면 됩니다. Ollama, LM Studio, llama.cpp, vLLM, 그리고 이 폴더의 Q38(포트 1923)이 그 형식입니다.

점수는 각 벤치의 공개 채점 규칙을 따릅니다. `--limit`으로 돌린 값은 앞부분만 채점한 결과입니다. 이 숫자는 일부 구간 점수이며, 전체 리더보드 점수와는 별개입니다.

## 설치

```powershell
cd local-bench
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\local-bench list
```

아래 명령은 이 가상환경이 켜져 있다고 가정합니다. 그렇지 않으면 `.\.venv\Scripts\local-bench`로 실행합니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

처음 실행하면 Hugging Face와 공식 저장소에서 문제 파일을 받아 `~/.cache/local-bench`와 Hugging Face 캐시에 저장합니다. 시험 문항은 이 저장소에 넣지 않습니다. BBH의 3-shot 예시만 채점 프롬프트로 포함합니다. GPQA 저자들은 문항을 웹에 그대로 올리지 말 것을 요청합니다.

키는 저장소에 넣지 않습니다. 로컬 서버는 API 키 없이 동작합니다. 원격 서버나 캐시 위치는 셸 환경 변수로 지정합니다. 프로그램을 시작한 폴더에 `.env`가 있으면, 셸에 아직 없는 이름만 그 파일에서 채웁니다. 빈 값은 건너뜁니다. 이름은 `.env.example`에 있고, `.env`는 git에서 제외됩니다.

| 변수 | 역할 |
|---|---|
| `OPENAI_API_KEY` | 원격 서버용 키입니다. 비우면 `local`입니다 |
| `LOCAL_BENCH_BASE_URL` | 모델 탐색에 추가할 OpenAI 호환 주소입니다 |
| `LOCAL_BENCH_CACHE` | 문제 캐시 디렉터리입니다. 기본값은 `~/.cache/local-bench`입니다 |

## 모델 고르기

```powershell
local-bench models
local-bench list
local-bench run gpqa
```

`run`에서 `--model`을 빼면, 이미 켜져 있는 서버의 모델 번호를 물어봅니다.

```powershell
local-bench run gpqa --base-url http://127.0.0.1:1923/v1 --model Qwen3.8-Flash-Next-NVFP4 --limit 5
local-bench run knowledge --model llama3.1:8b --limit 20
local-bench report
```

결과는 `runs/<모델>/<벤치>.jsonl`과 `.summary.json`에 쌓입니다. 같은 명령을 다시 실행하면 이미 받은 문항은 건너뜁니다. 처음부터 다시 받으려면 `--fresh`를 사용합니다.

로컬 GPU는 요청을 한 번에 하나씩 처리하는 경우가 많습니다. 기본값은 `--workers 1`입니다. 추론 모델은 `--max-tokens`를 늘립니다. 서버가 지원하면 `--no-think`로 사고 토큰을 끕니다.

## 실행하는 벤치

| 묶음 | 벤치 | 채점 |
|---|---|---|
| 과학 | `gpqa`, `gpqa-main`, `gpqa-extended` | simple-evals 0-shot CoT입니다. 선택지는 seed 0으로 섞습니다. `--repeats 4`가 simple-evals 평균입니다 |
| 일반 지식 | `mmlu`, `mmlu-pro`, `kmmlu` | MMLU는 생성형 5-shot입니다. MMLU-Pro는 0-shot CoT이며, 논문 설정은 `--shots 5`입니다. KMMLU는 한국어입니다 |
| 과학·상식 | `arc-challenge`, `arc-easy`, `openbookqa`, `commonsenseqa`, `boolq`, `winogrande`, `truthfulqa`, `siqa` | 마지막 `Answer:` 줄입니다. 테스트 정답이 비공개인 세트는 validation을 사용합니다 |
| 상식(생성형) | `hellaswag`, `piqa` | 모델이 보기를 고릅니다. 공식 표는 로그확률을 쓰므로 프로토콜이 다릅니다 |
| 수학 | `gsm8k`, `math500`, `aime2024`, `mgsm` | GSM8K는 8-shot CoT입니다. MATH-500과 AIME는 `\boxed{}`를 사용합니다. MATH-500은 문자열과 실수만 비교합니다. MGSM은 `--lang en`을 받습니다 |
| 코드 | `humaneval`, `mbpp` | pass@1입니다. 모델이 작성한 파이썬을 이 PC에서 실행합니다 |
| 지시 | `ifeval` | 대표 점수는 prompt-level strict입니다. instruction-level strict와 loose도 함께 나옵니다 |
| 추론·독해 | `bbh`, `musr`, `drop` | BBH는 Suzgun 3-shot CoT 뒤의 정확 일치입니다. MuSR은 객관식입니다. DROP은 토큰 F1입니다 |
| 에이전트 | `bfcl` | BFCL v4 파이썬 prompting 트랙의 AST 정확도입니다 |
| 아레나 | `arena-hard` | Arena-Hard-Auto v2입니다. 기준 모델과 심판 모델이 필요합니다 |

스위트:

- `quick` — GPQA Diamond, HumanEval, AIME 2024, OpenBookQA, IFEval
- `core` — 위의 짧은 목록에 전체 MMLU, GSM8K, ARC-Challenge, BFCL을 더합니다
- `knowledge`, `science`, `math`, `code`, `instruction`, `reasoning`, `reading`, `agent`, `arena`
- `aa-public` — Artificial Analysis가 사용해 온 공개 과제입니다. GPQA, MMLU-Pro, GSM8K, MATH-500, AIME, HumanEval, IFEval, BFCL
- `all` — `arena-hard`를 제외한 전부입니다

```powershell
local-bench run bfcl --category simple_python --limit 20
local-bench run mgsm --lang en --limit 20
local-bench run mmlu --subject abstract_algebra
local-bench run arena-hard --model my-model --baseline-model baseline --judge-model judge
```

## 리더보드 숫자와 맞는 부분

Artificial Analysis Intelligence Index v4.3의 가중치 상당수는 비공개 세트에 있습니다. AA-Briefcase, AutomationBench-AA, AA-Omniscience, CritPt는 문제가 공개되어 있지 않아 이 실행기에서 재현하지 않습니다. Terminal-Bench와 SciCode는 공개되어 있지만 도커와 과학 코드 샌드박스가 필요하며, 이 실행기의 범위 밖입니다. GPQA Diamond는 2026년 9월 지수에서 빠졌고, 새 모델에 대해 따로 공개하는 과학 점수로 남아 있습니다.

LM Arena Elo는 사람 투표에서 나옵니다. 오프라인 실행으로는 그 Elo를 만들 수 없습니다. `arena-hard`는 LM Arena가 공개한 질문과 심판 프롬프트로 그 순위를 근사합니다. 공개 표는 o3-mini 기준 답과 GPT-4.1 심판에 묶여 있습니다. 여기서 0.5는 지금 고른 기준 모델과 비등하다는 뜻입니다.

MMLU 원 논문과 HellaSwag, PIQA의 공식 점수는 다음 토큰 로그확률입니다. 로컬 채팅 서버는 그 확률을 돌려주지 않는 경우가 많아서, 이 실행기는 모델이 고른 문자를 채점합니다.

BFCL 점수는 파이썬 prompting 트랙입니다. 기본 범주는 simple, multiple, parallel, parallel_multiple이고, 점수는 범주 점수의 평균입니다. `--category all`에는 live와 irrelevance가 들어갑니다. 멀티턴 상태 환경, Java, JavaScript는 공식 BFCL 저장소가 담당합니다.

HumanEval과 MBPP는 생성한 코드를 서브프로세스에서 실행합니다. 시간 제한이 있고 `os.kill`, `os.system` 같은 함수 몇 개를 비워 둡니다. 이것은 보안 샌드박스가 아닙니다. `importlib.reload(os)`를 쓰면 그 차단은 바로 되돌아갑니다. 파일 읽기, 네트워크, `os.startfile`, `os.spawn*`은 열려 있습니다. 신뢰하는 모델만, 지워도 되는 환경에서 실행합니다.

IFEval의 문장 수 검사는 NLTK punkt 대신 같은 파일의 정규식 분할을 사용합니다. `number_sentences` 항목만 공식 스크립트와 조금 다를 수 있습니다.

BBH 기본값은 Suzgun et al.이 공개한 과제별 3-shot CoT 프롬프트입니다. 예시를 앞에 붙이고 `A: Let's think step by step.`으로 끝냅니다. 채점기는 마지막 `the answer is` 또는 `Answer:` 뒤를 정규화합니다. 객관식에서는 `(B)`와 `B`를 같은 답으로 봅니다. `--shots 0`은 예시를 붙이지 않으므로, 그 실행은 논문의 BBH 숫자와 별개입니다. logical deduction과 tracking shuffled objects의 3개·5개·7개 변형은 공개 프롬프트 파일 하나를 공유합니다. 이 실행기는 그 파일의 3-object 예시를 공개된 그대로 사용합니다.

MATH-500은 기호 동치를 확인하지 않습니다. 공백과 `\dfrac` 같은 표기를 정리한 문자열이 같거나, 양쪽이 모두 실수로 읽힐 때 같은 답입니다. `\frac{1}{2}`와 `0.5`는 다른 답입니다. 이 점수는 sympy를 쓰는 공식 MATH 채점보다 낮게 나올 수 있습니다.

GPQA에서 `--limit`은 반복을 펼친 뒤 앞의 N개만 남깁니다. `--repeats 4 --limit 20`은 첫 반복의 앞 20문항이며, 네 번 평균과는 다른 숫자입니다.

응답에 `usage`가 있으면 결과 줄에 토큰 수를 저장하고, 요약에 생성 tok/s를 적습니다. 서버가 usage를 빼먹으면 그 칸은 비어 있습니다. HTTP 429 또는 503이고 `Retry-After`가 있으면 그 시간만큼 기다린 뒤 다시 시도합니다. 상한은 60초입니다.

원 코드는 MIT 라이선스입니다. IFEval 채점기, BFCL 채점기, BBH CoT 프롬프트는 Apache-2.0입니다. 출처는 `src/local_bench/third_party/NOTICE`에 있습니다.
