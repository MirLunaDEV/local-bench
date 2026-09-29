# local-bench

로컬 모델을 골라, Artificial Analysis와 LM Arena가 근거로 삼는 **공개** 벤치마크를 그 모델에 그대로 돌리는 실행기다. 서버는 OpenAI 호환 `POST /v1/chat/completions`만 되면 된다. Ollama, LM Studio, llama.cpp, vLLM, 이 폴더의 Q38(포트 1923)이 그 형식이다.

점수는 각 벤치의 공개 채점 규칙을 따른다. `--limit`으로 앞부분만 돌린 숫자는 리더보드 점수와 비교하는 값이 아니다.

## 설치

```powershell
cd local-bench
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\local-bench list
```

아래 명령은 이 가상환경이 활성화된 상태를 가정한다. 활성화하지 않으면 `.\.venv\Scripts\local-bench`로 호출한다.

```powershell
.\.venv\Scripts\Activate.ps1
```

처음 실행할 때 문제 파일을 Hugging Face와 공식 저장소에서 받아 `~/.cache/local-bench`와 Hugging Face 캐시에 둔다. 시험 문항은 저장소에 넣지 않는다. BBH의 3-shot 예시만 채점 프롬프트로 포함한다. GPQA 저자들은 문항을 웹에 그대로 올리지 말 것을 요청한다.

키는 저장소에 넣지 않는다. 로컬 서버는 API 키 없이 된다. 원격 서버나 캐시 위치는 셸 환경 변수로 지정한다. 실행을 시작한 폴더에 `.env`가 있으면, 셸에 아직 없는 이름만 그 파일에서 채운다. 빈 칸은 건너뛴다. `.env.example`에 이름이 있고, `.env`는 git에서 빠진다.

| 변수 | 역할 |
|---|---|
| `OPENAI_API_KEY` | 원격 서버용 키. 비우면 `local` |
| `LOCAL_BENCH_BASE_URL` | 모델 탐색에 추가할 OpenAI 호환 주소 |
| `LOCAL_BENCH_CACHE` | 문제 캐시 디렉터리. 기본은 `~/.cache/local-bench` |

## 모델 고르기

```powershell
local-bench models
local-bench list
local-bench run gpqa
```

`run`에서 `--model`을 빼면, 터미널에서 켜져 있는 서버의 모델 번호를 물어본다.

```powershell
local-bench run gpqa --base-url http://127.0.0.1:1923/v1 --model Qwen3.8-Flash-Next-NVFP4 --limit 5
local-bench run knowledge --model llama3.1:8b --limit 20
local-bench report
```

결과는 `runs/<모델>/<벤치>.jsonl`과 `.summary.json`에 쌓인다. 같은 명령을 다시 실행하면 이미 받은 문항은 건너뛴다. 처음부터 다시 받으려면 `--fresh`.

로컬 GPU는 요청을 한 번에 하나 처리하는 경우가 많다. 기본 `--workers 1`. 추론 모델은 `--max-tokens`를 늘리고, 서버가 지원하면 `--no-think`로 사고 토큰을 끌 수 있다.

## 무엇을 돌리나

| 묶음 | 벤치 | 채점 |
|---|---|---|
| 과학 | `gpqa`, `gpqa-main`, `gpqa-extended` | simple-evals 0-shot CoT. 선택지는 seed 0으로 섞는다. `--repeats 4`가 simple-evals 평균 |
| 일반지식 | `mmlu`, `mmlu-pro`, `kmmlu` | MMLU는 생성형 5-shot. MMLU-Pro는 0-shot CoT, 논문 설정은 `--shots 5`. KMMLU는 한국어 |
| 과학·상식 | `arc-challenge`, `arc-easy`, `openbookqa`, `commonsenseqa`, `boolq`, `winogrande`, `truthfulqa`, `siqa` | 마지막 `Answer:` 줄. 테스트 정답이 비공개인 세트는 validation |
| 상식(생성형) | `hellaswag`, `piqa` | 보기를 고르는 점수. 공식표의 로그확률 점수와는 다른 프로토콜 |
| 수학 | `gsm8k`, `math500`, `aime2024`, `mgsm` | GSM8K는 8-shot CoT. MATH-500과 AIME는 `\boxed{}`. MATH-500은 문자열·실수 비교만 한다. MGSM은 `--lang en` |
| 코드 | `humaneval`, `mbpp` | pass@1. 모델이 쓴 파이썬을 이 PC에서 실행 |
| 지시 | `ifeval` | prompt-level strict가 대표 점수. instruction-level strict/loose도 같이 나온다 |
| 추론·독해 | `bbh`, `musr`, `drop` | BBH는 Suzgun 3-shot CoT 뒤 정확 일치. MuSR 객관식, DROP 토큰 F1 |
| 에이전트 | `bfcl` | BFCL v4 파이썬 prompting의 AST 정확도 |
| 아레나 | `arena-hard` | Arena-Hard-Auto v2. 기준 모델과 심판 모델이 필요하다 |

스위트:

- `quick` — GPQA Diamond, HumanEval, AIME 2024, OpenBookQA, IFEval
- `core` — 위에서 고른 것 + MMLU 전체 + GSM8K + ARC-Challenge + BFCL
- `knowledge`, `science`, `math`, `code`, `instruction`, `reasoning`, `reading`, `agent`, `arena`
- `aa-public` — Artificial Analysis가 공개 과제로 써 온 축: GPQA, MMLU-Pro, GSM8K, MATH-500, AIME, HumanEval, IFEval, BFCL
- `all` — `arena-hard`를 뺀 전부

```powershell
local-bench run bfcl --category simple_python --limit 20
local-bench run mgsm --lang en --limit 20
local-bench run mmlu --subject abstract_algebra
local-bench run arena-hard --model 내모델 --baseline-model 기준모델 --judge-model 심판모델
```

## 리더보드 숫자와 같은 것, 다른 것

Artificial Analysis Intelligence Index v4.3의 가중치 상당수는 비공개 세트다. AA-Briefcase, AutomationBench-AA, AA-Omniscience, CritPt는 문제가 공개되어 있지 않아 여기서 재현하지 않는다. Terminal-Bench와 SciCode는 공개지만 도커와 과학 코드 실행 환경이 필요하고, 이 실행기 범위 밖이다. GPQA Diamond는 2026년 9월 지수에서 빠졌고, 새 모델에 대해 따로 공개하는 과학 점수로 남아 있다.

LM Arena Elo는 사람 투표다. 오프라인으로 같은 Elo를 만들 수 없다. `arena-hard`는 그 순위를 자동으로 근사하려고 LM Arena가 공개한 질문과 심판 프롬프트다. 공개 표의 점수는 o3-mini 기준 답과 GPT-4.1 심판에 묶여 있다. 이 도구의 0.5는 "지금 고른 기준 모델과 비등"이다.

MMLU 원 논문과 HellaSwag, PIQA의 공식 점수는 다음 토큰 로그확률이다. 로컬 채팅 서버에는 그 확률이 없는 경우가 많아서, 여기서는 모델이 고른 문자를 채점한다.

BFCL 점수는 파이썬 prompting 트랙이다. 기본 범주는 simple, multiple, parallel, parallel_multiple이고 범주 점수의 평균을 낸다. `--category all`에 live와 irrelevance가 들어간다. 멀티턴 상태 환경, Java, JavaScript는 공식 BFCL 저장소가 담당한다.

HumanEval과 MBPP는 생성 코드를 서브프로세스에서 실행한다. 시간 제한이 있고 `os.kill`, `os.system` 같은 함수 몇 개를 비워 두지만, 이것은 보안 샌드박스가 아니다. `importlib.reload(os)`로 그 차단은 바로 되돌려진다. 파일 읽기, 네트워크, `os.startfile`, `os.spawn*`은 열려 있다. 믿는 모델만, 지워도 되는 환경에서 돌린다.

IFEval의 문장 수 검사는 NLTK punkt 대신 같은 파일에 있는 정규식 분할을 쓴다. `number_sentences` 항목만 공식 스크립트와 조금 다를 수 있다.

BBH 기본값은 Suzgun et al.이 공개한 과제별 3-shot CoT 프롬프트다. 문제 앞에 그 예시를 붙이고 `A: Let's think step by step.`으로 끝낸 뒤, 응답에서 마지막 `the answer is` 또는 `Answer:` 뒤를 정규화해 정답과 비교한다. 객관식은 `(B)`와 `B`를 같은 답으로 본다. `--shots 0`은 예시를 붙이지 않으므로 논문의 BBH 숫자와 비교하는 값이 아니다. logical deduction과 tracking shuffled objects의 3개·5개·7개 변형은 공개 프롬프트 파일이 서로 같다. 그 파일의 3-object 예시를 그대로 쓴다.

MATH-500은 기호 동치를 풀지 않는다. 공백과 `\dfrac` 같은 표기를 정리한 문자열이 같거나, 양쪽이 모두 실수로 읽힐 때만 맞다. `\frac{1}{2}`와 `0.5`는 다른 답이다. 그래서 이 점수는 sympy를 쓰는 공식 MATH 채점보다 낮게 나올 수 있다.

GPQA에서 `--limit`은 반복을 펼친 뒤의 앞에서 N개만 남긴다. `--repeats 4 --limit 20`은 네 번 평균이 아니라 첫 반복의 앞 20문항이다.

응답에 `usage`가 있으면 결과 줄에 토큰 수를 저장하고, 요약에 생성 tok/s를 적는다. 서버가 usage를 빼먹으면 그 칸은 비어 있다. 호출이 429 또는 503이고 `Retry-After`가 있으면 그 시간만큼 기다린 뒤 다시 시도한다. 상한은 60초다.

원 코드는 MIT 라이선스다. IFEval, BFCL 채점기, BBH CoT 프롬프트는 Apache-2.0이고, 출처는 `src/local_bench/third_party/NOTICE`에 있다.
