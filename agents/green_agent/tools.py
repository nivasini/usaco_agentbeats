# -*- coding: utf-8 -*-
"""
USACO Green Agent - Competitive Programming Evaluator
Evaluates Python solutions against test cases with time limits.
"""

import os
import sys
import json
import tempfile
import subprocess
import time
from pathlib import Path
try:
    import agentbeats as ab
except Exception:  # Fallback shim for local testing without agentbeats installed
    class _Shim:
        def tool(self, fn=None, **kwargs):
            if fn:
                return fn

            def deco(f):
                return f

            return deco

    ab = _Shim()

TIME_LIMIT_SECONDS = 4.0

# Pick the problem you want the green agent to serve/evaluate.
# You can override via the USACO_PROBLEM_ID env var.
DEFAULT_PROBLEM_ID = "735_bronze_the_lost_cow"
PROBLEM_ID = os.environ.get("USACO_PROBLEM_ID", DEFAULT_PROBLEM_ID)

# Get the directory where this file is located, then go up to project root
# agents/green_agent/tools.py -> parent.parent.parent = project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_QUESTIONS = _PROJECT_ROOT / "final_data_subset" / "usaco_questions.json"
_DEFAULT_TESTS = _PROJECT_ROOT / "final_data_subset" / "tests"
QUESTIONS_PATH = Path(os.environ.get("USACO_QUESTIONS_PATH", _DEFAULT_QUESTIONS))
TESTS_ROOT = Path(os.environ.get("USACO_TESTS_ROOT", _DEFAULT_TESTS))


def _load_problem(problem_id: str) -> dict:
    with QUESTIONS_PATH.open("r") as f:
        data = json.load(f)
    if problem_id not in data:
        raise ValueError(f"Problem id {problem_id!r} not found in {QUESTIONS_PATH}")
    return data[problem_id]


_PROBLEM_DATA = _load_problem(PROBLEM_ID)
_SAMPLES = _PROBLEM_DATA.get("samples", []) or [{"input": "", "output": ""}]

PROBLEM_STATEMENT = _PROBLEM_DATA.get("description") or _PROBLEM_DATA.get("description_no_samples") or ""
SAMPLE_INPUT = (_SAMPLES[0].get("input") or "").rstrip() + "\n"
SAMPLE_OUTPUT = (_SAMPLES[0].get("output") or "").rstrip() + "\n"


@ab.tool
def get_problem() -> str:
    """
    Get the USACO problem statement, input/output format, and constraints.
    """
    return PROBLEM_STATEMENT


@ab.tool
def get_sample_test() -> dict:
    """
    Get the sample test case for the problem.
    Returns a dict with 'input' and 'expected_output'.
    """
    return {
        "input": SAMPLE_INPUT,
        "expected_output": SAMPLE_OUTPUT
    }


@ab.tool
def evaluate_solution(code: str) -> dict:
    """
    Evaluate a Python solution against all test cases.
    
    Args:
        code: Complete Python 3 program as a string
        
    Returns:
        dict with evaluation results including pass/fail, time, and details
    """
    results = {
        "sample_test": _run_single_test(code, SAMPLE_INPUT, SAMPLE_OUTPUT, "Sample"),
    }

    problem_dir = TESTS_ROOT / PROBLEM_ID
    pairs = _find_test_pairs(problem_dir)

    for idx, (in_path, out_path) in enumerate(pairs, 1):
        with open(in_path, "r", encoding="utf-8") as fin:
            input_text = fin.read()
        with open(out_path, "r", encoding="utf-8") as fout:
            expected_output = fout.read()
        name = f"test_{idx}"
        results[name] = _run_single_test(code, input_text, expected_output, name)
    
    all_passed = all(r["passed"] for r in results.values())

    # Build summary stats without logging per-test IO
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r["passed"])
    failed_tests = total_tests - passed_tests
    failure_ids = {}

    def _classify_failure(res: dict) -> str:
        if res.get("error") == "Time limit exceeded" or not res.get("time_limit_ok", True):
            return "time_limit"
        if not res.get("exit_code_ok", True):
            return "runtime_error"
        if not res.get("output_matches", True):
            actual = res.get("actual_output", "").strip() if isinstance(res.get("actual_output"), str) else ""
            if actual == "":
                return "empty_output"
            return "wrong_answer"
        return "unknown"

    reasons = {}
    for res in results.values():
        if res["passed"]:
            continue
        reason = _classify_failure(res)
        reasons[reason] = reasons.get(reason, 0) + 1
        failure_ids.setdefault(reason, []).append(res.get("test_name"))
    
    return {
        "verdict": "ACCEPTED" if all_passed else "REJECTED",
        "all_tests_passed": all_passed,
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "failures_by_reason": reasons,
            "failure_ids_by_reason": failure_ids,
        },
        "details": results
    }


def _find_test_pairs(problem_dir: Path):
    """
    Discover (input_file, output_file) pairs in a problem test directory.
    Supports:
      - *.in / *.out   (e.g., 1.in ↔ 1.out)
      - I.x / O.x      (e.g., I.1 ↔ O.1)
    """
    if not problem_dir.is_dir():
        raise FileNotFoundError(f"Problem test directory not found: {problem_dir}")

    entries = sorted(
        f for f in os.listdir(problem_dir)
        if not f.startswith(".") and (problem_dir / f).is_file()
    )

    pairs = []
    for fname in entries:
        # Pattern 1: *.in / *.out
        if fname.endswith(".in"):
            base = fname[:-3]
            out_name = base + ".out"
            in_path = problem_dir / fname
            out_path = problem_dir / out_name
            if out_path.exists():
                pairs.append((in_path, out_path))

        # Pattern 2: I.x / O.x
        if fname.startswith("I."):
            suffix = fname[2:]
            out_name = "O." + suffix
            in_path = problem_dir / fname
            out_path = problem_dir / out_name
            if out_path.exists():
                pairs.append((in_path, out_path))

    seen = set()
    unique_pairs = []
    for in_path, out_path in pairs:
        key = (in_path.name, out_path.name)
        if key not in seen:
            seen.add(key)
            unique_pairs.append((in_path, out_path))

    unique_pairs.sort(key=lambda p: p[0].name)
    return unique_pairs


def _run_single_test(code: str, input_text: str, expected_output: str, test_name: str) -> dict:
    """
    Run code with given input and check against expected output.
    """
    tmpdir = tempfile.mkdtemp(prefix="usaco_eval_")
    code_file = os.path.join(tmpdir, "solution.py")
    
    try:
        # Write code to file
        with open(code_file, "w") as f:
            f.write(code)
        
        # Run with timeout
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, code_file],
            input=input_text.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TIME_LIMIT_SECONDS + 0.5
        )
        elapsed = time.perf_counter() - start
        
        # Check results
        output = proc.stdout.decode()
        stderr = proc.stderr.decode()
        
        output_matches = _normalize_output(output) == _normalize_output(expected_output)
        time_ok = elapsed <= TIME_LIMIT_SECONDS
        exit_ok = proc.returncode == 0
        
        passed = output_matches and time_ok and exit_ok
        
        result = {
            "test_name": test_name,
            "passed": passed,
            "time_seconds": round(elapsed, 3),
            "time_limit_ok": time_ok,
            "exit_code_ok": exit_ok,
            "output_matches": output_matches
        }
        
        if not passed:
            result["stderr"] = stderr if stderr else None
            if not output_matches:
                result["expected_output"] = expected_output.strip()
                result["actual_output"] = output.strip()
        
        return result
        
    except subprocess.TimeoutExpired:
        return {
            "test_name": test_name,
            "passed": False,
            "time_seconds": TIME_LIMIT_SECONDS,
            "time_limit_ok": False,
            "error": "Time limit exceeded"
        }
    except Exception as e:
        return {
            "test_name": test_name,
            "passed": False,
            "error": str(e)
        }
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _normalize_output(s: str) -> str:
    """Normalize output for comparison."""
    return "\n".join(line.rstrip() for line in s.strip().splitlines())
