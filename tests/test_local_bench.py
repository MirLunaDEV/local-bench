import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_bench.catalog import SUITES, all_tasks, resolve_targets
from local_bench.client import _retry_delay
from local_bench.runner import JsonlStore, _ordered_rows, run_task
from local_bench.sandbox import run_python
from local_bench.scoreutil import (
    extract_boxed,
    extract_last_number,
    extract_letter,
    math_equal,
    numbers_equal,
    short_answer,
    token_f1,
    visible_text,
)
from local_bench.tasks.base import Example, Task
from local_bench.tasks.bfcl_decode import decode_calls
from local_bench.tasks.code_tasks import extract_code, grade_code
from local_bench.tasks.gpqa import grade_gpqa
from local_bench.third_party.bfcl.ast_checker import Language, ast_checker


class ScoreTests(unittest.TestCase):
    def test_visible_text_strips_think(self):
        text = "<think>Answer: A is tempting</think>\nAnswer: C"
        self.assertEqual(extract_letter(visible_text(text)), "C")

    def test_last_answer_wins(self):
        text = "Answer: A\nWait.\nAnswer: D"
        self.assertEqual(extract_letter(text), "D")

    def test_gsm_number(self):
        text = "working\n#### 1,024"
        self.assertEqual(extract_last_number(text), "1024")
        self.assertTrue(numbers_equal("1024", "1024.0"))

    def test_boxed(self):
        text = r"reason \boxed{1} later \boxed{\frac{1}{2}}"
        self.assertEqual(extract_boxed(text), r"\frac{1}{2}")
        self.assertTrue(math_equal(r"\dfrac{1}{2}", r"\frac{1}{2}"))
        self.assertFalse(math_equal(r"\frac{1}{2}", "0.5"))

    def test_article_is_not_a_choice(self):
        self.assertIsNone(extract_letter("the answer is a good explanation of the plot."))
        self.assertEqual(extract_letter("the answer is a good explanation.\nAnswer: C"), "C")
        self.assertEqual(extract_letter("So the answer is (A)."), "A")

    def test_short_answer_without_colon(self):
        self.assertEqual(short_answer("Working.\nSo the answer is (B)."), "(B).")
        self.assertEqual(short_answer("Answer: 7"), "7")
        self.assertEqual(short_answer("So the answer is True.\nSo the answer is False."), "False.")

    def test_token_f1(self):
        self.assertEqual(token_f1("the 7 cats", "7 cats"), 1.0)
        self.assertEqual(token_f1("nope", "7 cats"), 0.0)


class BfclTests(unittest.TestCase):
    def test_triangle_optional_unit(self):
        functions = [
            {
                "name": "calculate_triangle_area",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base": {"type": "integer"},
                        "height": {"type": "integer"},
                        "unit": {"type": "string"},
                    },
                    "required": ["base", "height"],
                },
            }
        ]
        gold = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
        calls = decode_calls("[calculate_triangle_area(base=10, height=5)]", functions)
        result = ast_checker(functions, calls, gold, Language.PYTHON, "simple_python", "local")
        self.assertTrue(result["valid"], result)

    def test_wrong_value_fails(self):
        functions = [
            {
                "name": "calculate_triangle_area",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base": {"type": "integer"},
                        "height": {"type": "integer"},
                    },
                    "required": ["base", "height"],
                },
            }
        ]
        gold = [{"calculate_triangle_area": {"base": [10], "height": [5]}}]
        calls = decode_calls("```python\n[calculate_triangle_area(base=9, height=5)]\n```", functions)
        result = ast_checker(functions, calls, gold, Language.PYTHON, "simple_python", "local")
        self.assertFalse(result["valid"])

    def test_parallel_order_free(self):
        functions = [
            {
                "name": "get_a",
                "parameters": {"type": "dict", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
            },
            {
                "name": "get_b",
                "parameters": {"type": "dict", "properties": {"y": {"type": "string"}}, "required": ["y"]},
            },
        ]
        gold = [{"get_a": {"x": [1]}}, {"get_b": {"y": ["Hi"]}}]
        calls = decode_calls('[get_b(y="hi"), get_a(x=1)]', functions)
        result = ast_checker(functions, calls, gold, Language.PYTHON, "parallel", "local")
        self.assertTrue(result["valid"], result)

    def test_dotted_name(self):
        functions = [
            {
                "name": "math.factorial",
                "parameters": {"type": "dict", "properties": {"number": {"type": "integer"}}, "required": ["number"]},
            }
        ]
        gold = [{"math.factorial": {"number": [5]}}]
        calls = decode_calls("[math.factorial(number=5)]", functions)
        result = ast_checker(functions, calls, gold, Language.PYTHON, "simple_python", "local")
        self.assertTrue(result["valid"], result)


class CodeTests(unittest.TestCase):
    def test_sandbox_pass_and_fail(self):
        ok, _ = run_python("assert 1 + 1 == 2\n")
        self.assertTrue(ok)
        bad, detail = run_python("assert 1 == 2\n")
        self.assertFalse(bad)
        self.assertIn("AssertionError", detail)

    def test_sandbox_timeout(self):
        ok, detail = run_python("import time\ntime.sleep(30)\n", timeout=1)
        self.assertFalse(ok)
        self.assertEqual(detail, "timed out")

    def test_humaneval_shape(self):
        problem = {
            "prompt": "def add(a, b):\n    \"\"\"Add.\"\"\"\n",
            "entry_point": "add",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
        }
        example = Example(
            id="HumanEval/0",
            messages=[],
            gold="HumanEval/0",
            meta={"kind": "humaneval", "problem": problem, "timeout": 5},
        )
        text = "```python\ndef add(a, b):\n    return a + b\n```"
        self.assertIn("def add", extract_code(text))
        graded = grade_code(example, text)
        self.assertEqual(graded["score"], 1.0)


class IfevalTests(unittest.TestCase):
    def test_comma_and_keyword(self):
        from local_bench.tasks.ifeval_task import grade_ifeval

        example = Example(
            id="1",
            messages=[],
            gold="1",
            meta={
                "prompt": "Say hello.",
                "instruction_ids": ["punctuation:no_comma", "keywords:existence"],
                "kwargs": [{}, {"keywords": ["alpha"]}],
            },
        )
        good = grade_ifeval(example, "alpha is here")
        self.assertEqual(good["prompt_strict"], 1.0)
        bad = grade_ifeval(example, "alpha, beta")
        self.assertEqual(bad["prompt_strict"], 0.0)
        self.assertEqual(bad["inst_strict"], 0.5)


class CatalogTests(unittest.TestCase):
    def test_ids_and_suites(self):
        tasks = all_tasks()
        ids = [task.id for task in tasks]
        self.assertEqual(len(ids), len(set(ids)))
        for suite, members in SUITES.items():
            missing = [item for item in members if item not in ids]
            self.assertEqual(missing, [], suite)
        resolved = resolve_targets("quick")
        self.assertEqual([task.id for task in resolved], SUITES["quick"])
        self.assertNotIn("arena-hard", [task.id for task in resolve_targets("all")])


class BbhTests(unittest.TestCase):
    def test_prompts_are_official_three_shot(self):
        from local_bench.tasks.reasoning import cot_prompts, render_bbh_prompt

        prompts = cot_prompts()
        self.assertEqual(len(prompts), 27)
        for name, (instruction, examples) in prompts.items():
            self.assertTrue(instruction.strip(), name)
            self.assertEqual(len(examples), 3, name)
            for example in examples:
                self.assertTrue(example.startswith("Q:"), name)
                self.assertIn("So the answer is", example)
        text = render_bbh_prompt("boolean_expressions", "True and True is", 3)
        self.assertEqual(text.count("So the answer is"), 3)
        self.assertTrue(text.endswith("Q: True and True is\nA: Let's think step by step."))
        self.assertNotIn("canary GUID", text)
        self.assertEqual(render_bbh_prompt("boolean_expressions", "True and True is", 0), "True and True is")
        three = cot_prompts()["logical_deduction_three_objects"]
        self.assertEqual(three, cot_prompts()["logical_deduction_five_objects"])
        self.assertEqual(three, cot_prompts()["logical_deduction_seven_objects"])

    def test_answer_is_scores(self):
        from local_bench.tasks.reasoning import grade_bbh

        def grade(gold, text):
            example = Example(id="1", messages=[], gold=gold, meta={})
            return grade_bbh(example, text)

        self.assertEqual(grade("(B)", "Working.\nSo the answer is (B).")["score"], 1.0)
        self.assertEqual(grade("(B)", "So the answer is (B). This matches the second option.")["score"], 1.0)
        self.assertEqual(grade("(C)", "So the answer is C")["score"], 1.0)
        self.assertEqual(grade("False", "So the answer is True.\nSo the answer is False.")["score"], 1.0)
        self.assertEqual(grade("yes", "So the answer is yes.")["score"], 1.0)
        self.assertEqual(grade("-219", "So the answer is -219.")["score"], 1.0)
        self.assertEqual(grade("costume counterpart oven", "So the answer is costume counterpart oven.")["score"], 1.0)
        self.assertEqual(grade("] } ]", "So the answer is ] } ].")["score"], 1.0)
        self.assertEqual(grade("(A)", "I never commit.")["score"], 0.0)


class RunnerTests(unittest.TestCase):
    def test_fake_model_resume(self):
        class Fake:
            model = "fake-model"
            base_url = "http://127.0.0.1:9/v1"

            def chat(self, messages):
                return "Answer: B"

        task = Task(
            id="toy",
            title="Toy",
            group="test",
            protocol="toy",
            build=lambda opts: [
                Example(id="q1", messages=[{"role": "user", "content": "hi"}], gold="B", meta={"allowed": "ABCD"})
            ],
            grade_fn=grade_gpqa,
        )

        class Opts:
            limit = None
            shots = None
            subject = None
            fresh = False
            workers = 1
            max_errors = 3

        with tempfile.TemporaryDirectory() as directory:
            summary = run_task(task, Fake(), Opts(), Path(directory))
            self.assertEqual(summary["metrics"]["score"], 1.0)
            again = run_task(task, Fake(), Opts(), Path(directory))
            self.assertEqual(again["metrics"]["n"], 1)
            saved = json.loads(Path(summary["path"]).read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(saved["gold"], "B")

    def test_truncated_jsonl_is_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bbh.jsonl"
            path.write_text('{"id": "q1", "gold": "B"}\n{"id": "q2", "gold":\n', encoding="utf-8")
            store = JsonlStore(path)
            self.assertEqual(list(store.rows), ["q1"])
            reloaded = path.read_text(encoding="utf-8")
            self.assertNotIn("q2", reloaded)
            self.assertIn("q1", reloaded)
            again = JsonlStore(path)
            self.assertEqual(list(again.rows), ["q1"])

    def test_broken_middle_line_still_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bbh.jsonl"
            path.write_text('{"id": "q1"}\nNOT JSON\n{"id": "q2"}\n', encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                JsonlStore(path)

    def test_dataset_order_and_usage(self):
        class Reply:
            text = "Answer: B"
            usage = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}

        class Fake:
            model = "fake-model"
            base_url = "http://127.0.0.1:9/v1"

            def complete(self, messages):
                time.sleep(0.02)
                return Reply()

        task = Task(
            id="toy",
            title="Toy",
            group="test",
            protocol="toy",
            build=lambda opts: [
                Example(id="q2", messages=[{"role": "user", "content": "hi"}], gold="B", meta={"allowed": "ABCD"}),
                Example(id="q1", messages=[{"role": "user", "content": "hi"}], gold="B", meta={"allowed": "ABCD"}),
            ],
            grade_fn=grade_gpqa,
        )

        class Opts:
            limit = None
            shots = None
            subject = None
            fresh = False
            workers = 1
            max_errors = 3

        with tempfile.TemporaryDirectory() as directory:
            summary = run_task(task, Fake(), Opts(), Path(directory))
            self.assertEqual(summary["metrics"]["score"], 1.0)
            self.assertIn("completion_tokens_per_s", summary)
            saved = Path(summary["path"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(saved[0])["usage"]["completion_tokens"], 4)
            ids = [json.loads(line)["id"] for line in saved]
            ordered = _ordered_rows(JsonlStore(Path(summary["path"])), task.examples(Opts()))
            self.assertEqual([row["id"] for row in ordered], ["q2", "q1"])
            self.assertEqual(ids, ["q2", "q1"])

    def test_retry_after(self):
        self.assertEqual(_retry_delay("12", 1.5), 12.0)
        self.assertEqual(_retry_delay("120", 1.5), 60.0)
        self.assertEqual(_retry_delay("soon", 1.5), 1.5)
        self.assertEqual(_retry_delay(None, 2.0), 2.0)


class EnvFileTests(unittest.TestCase):
    def test_fills_missing_and_keeps_existing(self):
        from local_bench.cli import load_local_env

        names = ("OPENAI_API_KEY", "LOCAL_BENCH_BASE_URL", "LOCAL_BENCH_CACHE", "LOCAL_BENCH_SHOULD_NOT")
        saved = {name: os.environ.get(name) for name in names}
        for name in names:
            os.environ.pop(name, None)
        os.environ["LOCAL_BENCH_CACHE"] = "already-set"
        os.environ["OPENAI_API_KEY"] = ""
        try:
            with tempfile.TemporaryDirectory() as directory:
                Path(directory, ".env").write_text(
                    "\n".join(
                        [
                            "# OPENAI_API_KEY=from-comment",
                            "OPENAI_API_KEY=",
                            "OPENAI_API_KEY=from-file",
                            "export LOCAL_BENCH_BASE_URL=\"http://127.0.0.1:11434/v1\"",
                            "LOCAL_BENCH_CACHE=from-file",
                            "LOCAL_BENCH_SHOULD_NOT=",
                            "not-a-name=1",
                            "BARE",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                load_local_env(Path(directory) / ".env")
            self.assertEqual(os.environ.get("OPENAI_API_KEY"), "")
            self.assertEqual(os.environ.get("LOCAL_BENCH_CACHE"), "already-set")
            self.assertEqual(os.environ.get("LOCAL_BENCH_BASE_URL"), "http://127.0.0.1:11434/v1")
            self.assertNotIn("LOCAL_BENCH_SHOULD_NOT", os.environ)
            self.assertNotIn("not-a-name", os.environ)
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_main_loads_env_from_cwd(self):
        from local_bench.cli import main

        name = "LOCAL_BENCH_CACHE"
        saved = os.environ.pop(name, None)
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                Path(directory, ".env").write_text("LOCAL_BENCH_CACHE=C:/from-dotenv\n", encoding="utf-8")
                os.chdir(directory)
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        code = main(["list"])
                    self.assertEqual(code, 0)
                    self.assertEqual(os.environ.get(name), "C:/from-dotenv")
                finally:
                    os.chdir(previous)
        finally:
            if saved is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = saved


if __name__ == "__main__":
    unittest.main()
