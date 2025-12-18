#!/usr/bin/env python3
"""
Test USACO agents by directly importing and calling their tools.
This simulates what happens during an AgentBeats assessment.
"""

import sys
import os
import argparse
import importlib.util
import json
import random
from pathlib import Path

# Local helper to mirror green's test discovery so we can log exact cases
def _find_test_pairs(problem_dir: Path):
    entries = sorted(
        f for f in os.listdir(problem_dir)
        if not f.startswith(".") and (problem_dir / f).is_file()
    )
    pairs = []
    for fname in entries:
        if fname.endswith(".in"):
            base = fname[:-3]
            out_name = base + ".out"
            in_path = problem_dir / fname
            out_path = problem_dir / out_name
            if out_path.exists():
                pairs.append((in_path, out_path))
        if fname.startswith("I."):
            suffix = fname[2:]
            out_name = "O." + suffix
            in_path = problem_dir / fname
            out_path = problem_dir / out_name
            if out_path.exists():
                pairs.append((in_path, out_path))
    seen = set()
    uniq = []
    for a, b in pairs:
        key = (a.name, b.name)
        if key not in seen:
            seen.add(key)
            uniq.append((a, b))
    uniq.sort(key=lambda p: p[0].name)
    return uniq

# Set up logging per problem id
def _init_logger(problem_id: str):
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"{problem_id}.txt"
    log_file = log_path.open("w", encoding="utf-8")

    def logprint(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        print(text, **kwargs)
        log_file.write(text)
        log_file.write("\n" if not text.endswith("\n") else "")
        log_file.flush()

    return logprint, log_file


def _load_agents():
    # Load green agent tools
    green_spec = importlib.util.spec_from_file_location(
        "green_tools",
        "agents/green_agent/tools.py"
    )
    green_tools = importlib.util.module_from_spec(green_spec)
    green_spec.loader.exec_module(green_tools)

    # Load white agent tools
    white_spec = importlib.util.spec_from_file_location(
        "white_tools",
        "agents/white_agent/tools.py"
    )
    white_tools = importlib.util.module_from_spec(white_spec)
    white_spec.loader.exec_module(white_tools)
    return green_tools, white_tools


def main():
    parser = argparse.ArgumentParser(description="Run local agent test harness.")
    parser.add_argument(
        "--data-root",
        help="Base folder containing usaco_questions.json and tests/ (default: final_data_subset)",
        default="final_data_subset",
    )
    parser.add_argument(
        "--questions-path",
        help="Override path to questions JSON (default: <data-root>/usaco_questions.json)",
    )
    parser.add_argument(
        "--tests-root",
        help="Override path to tests directory (default: <data-root>/tests)",
    )
    parser.add_argument(
        "--level",
        help="Select a random problem of the given level (bronze/silver/gold/platinum) from the questions file",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    questions_path = Path(args.questions_path) if args.questions_path else data_root / "usaco_questions.json"
    tests_root = Path(args.tests_root) if args.tests_root else data_root / "tests"

    # If level is provided, choose a random problem id of that level and set env
    if args.level:
        with questions_path.open("r") as f:
            qdata = json.load(f)
        candidates = [pid for pid, meta in qdata.items() if meta.get("problem_level") == args.level]
        if not candidates:
            raise SystemExit(f"No problems found at level '{args.level}' in {questions_path}")
        chosen_pid = random.choice(candidates)
        os.environ["USACO_PROBLEM_ID"] = chosen_pid

    # Propagate to green agent via env before import
    os.environ["USACO_QUESTIONS_PATH"] = str(questions_path)
    os.environ["USACO_TESTS_ROOT"] = str(tests_root)

    green_tools, white_tools = _load_agents()

    problem_id = getattr(green_tools, "PROBLEM_ID", "unknown_problem")
    logprint, log_file = _init_logger(problem_id)

    print("=" * 70)
    logprint("=" * 70)
    logprint("USACO Local Agent Test - Direct Python Calls")
    logprint("=" * 70)
    logprint()

    # Step 1: Green agent provides the problem
    logprint("Step 1: Green agent provides problem statement")
    logprint("-" * 70)
    problem = green_tools.get_problem()
    logprint(problem)
    logprint("-" * 70)
    logprint()

    # Step 1b: White agent receives the problem (so its state is set)
    if hasattr(white_tools, "request_problem_from_green_agent"):
        try:
            white_tools.request_problem_from_green_agent(problem)
        except Exception as e:
            logprint(f"(Warning) Failed to pass problem to white agent: {e}")
        logprint()

    # Step 2: Green agent provides sample test
    logprint("Step 2: Green agent provides sample test")
    logprint("-" * 70)
    sample = green_tools.get_sample_test()
    logprint(f"Input: {sample['input']}")
    logprint(f"Expected Output: {sample['expected_output']}")
    logprint("-" * 70)
    logprint()

    # Step 3: White agent generates solution
    logprint("Step 3: White agent generates solution")
    logprint("-" * 70)
    solution = white_tools.get_my_solution()
    # Log full solution (no truncation)
    logprint(solution)
    logprint("-" * 70)
    logprint()

    # Step 4: Green agent evaluates the solution
    logprint("Step 4: Green agent evaluates the solution")
    logprint("-" * 70)
    result = green_tools.evaluate_solution(solution)
    logprint(f"Verdict: {result['verdict']}")
    logprint(f"All tests passed: {result['all_tests_passed']}")
    if "summary" in result:
        summary = result["summary"]
        logprint(f"Summary: total={summary.get('total_tests')}, "
                 f"passed={summary.get('passed')}, failed={summary.get('failed')}, "
                 f"failures_by_reason={summary.get('failures_by_reason')}, "
                 f"failure_ids_by_reason={summary.get('failure_ids_by_reason')}")
    logprint()
    logprint("Test Details:")
    for test_name, test_result in result['details'].items():
        logprint(f"  {test_name}:")
        logprint(f"    Passed: {test_result['passed']}")
        if not test_result['passed']:
            if 'expected' in test_result:
                logprint(f"    Expected: {test_result['expected']}")
            if 'actual' in test_result:
                logprint(f"    Got: {test_result['actual']}")
            if test_result.get('error'):
                logprint(f"    Error: {test_result['error']}")
            # Print all available fields for debugging
            for key, value in test_result.items():
                if key not in ['passed', 'expected', 'actual', 'error']:
                    logprint(f"    {key}: {value}")
    logprint("-" * 70)
    logprint()

    # Step 5: Test with buggy solution
    logprint("Step 5: Testing with buggy solution")
    logprint("-" * 70)
    buggy_solution = white_tools.get_buggy_solution()
    logprint("Buggy solution:")
    logprint(buggy_solution)
    logprint()
    buggy_result = green_tools.evaluate_solution(buggy_solution)
    logprint(f"Verdict: {buggy_result['verdict']}")
    logprint(f"All tests passed: {buggy_result['all_tests_passed']}")
    if "summary" in buggy_result:
        summary = buggy_result["summary"]
        logprint(f"Summary: total={summary.get('total_tests')}, "
                 f"passed={summary.get('passed')}, failed={summary.get('failed')}, "
                 f"failures_by_reason={summary.get('failures_by_reason')}, "
                 f"failure_ids_by_reason={summary.get('failure_ids_by_reason')}")
    logprint()
    logprint("Test Details:")
    for test_name, test_result in buggy_result['details'].items():
        logprint(f"  {test_name}:")
        logprint(f"    Passed: {test_result['passed']}")
        if not test_result['passed']:
            if 'expected' in test_result:
                logprint(f"    Expected: {test_result['expected']}")
            if 'actual' in test_result:
                logprint(f"    Got: {test_result['actual']}")
            if test_result.get('error'):
                logprint(f"    Error: {test_result['error']}")
            # Print all available fields for debugging
            for key, value in test_result.items():
                if key not in ['passed', 'expected', 'actual', 'error']:
                    logprint(f"    {key}: {value}")
    logprint("-" * 70)
    logprint()

    # Summary
    logprint("=" * 70)
    logprint("SUMMARY")
    logprint("=" * 70)
    logprint(f"Correct solution: {result['verdict']}")
    if "summary" in result:
        summary = result["summary"]
        logprint(f"  -> total={summary.get('total_tests')}, "
                 f"passed={summary.get('passed')}, failed={summary.get('failed')}, "
                 f"failures_by_reason={summary.get('failures_by_reason')}, "
                 f"failure_ids_by_reason={summary.get('failure_ids_by_reason')}")
    logprint(f"Buggy solution: {buggy_result['verdict']}")
    if "summary" in buggy_result:
        summary = buggy_result["summary"]
        logprint(f"  -> total={summary.get('total_tests')}, "
                 f"passed={summary.get('passed')}, failed={summary.get('failed')}, "
                 f"failures_by_reason={summary.get('failures_by_reason')}, "
                 f"failure_ids_by_reason={summary.get('failure_ids_by_reason')}")
    logprint()
    logprint("This simulates what will happen when the assessment runs on AgentBeats:")
    logprint("1. Green agent (assessor) provides problem and evaluates solutions")
    logprint("2. White agent (assessee) attempts to solve the problem")
    logprint("3. Green agent checks if the solution is correct")
    logprint("=" * 70)

    log_file.close()

if __name__ == "__main__":
    main()
