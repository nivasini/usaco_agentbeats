#!/usr/bin/env python3
"""
Perturb a USACO problem and its test cases using an LLM.

Workflow:
1) Pick a problem (by id or random, optionally filtered by level).
2) Ask the LLM to produce:
   - A perturbed problem statement (logically equivalent)
   - A mapping of old->new narrative details
   - Perturbed sample I/O
   - Python code to transform test inputs/outputs
3) Apply the transformer to all tests for that problem.
4) Save results under perturbed_data/<new_problem_id>/ with tests in /tests.

Requirements:
- OPENAI_API_KEY in .env or env.local
- final_data/usaco_questions.json and final_data/tests/<problem_id> present
"""
import argparse
import json
import os
import random
import shutil
import textwrap
from pathlib import Path
from typing import Dict, Any, Callable

import dotenv
from litellm import completion

ROOT = Path(__file__).resolve().parent
dotenv.load_dotenv(ROOT / ".env")
dotenv.load_dotenv(ROOT / "env.local")

QUESTIONS_PATH = ROOT / "final_data" / "usaco_questions.json"
TESTS_ROOT = ROOT / "final_data" / "tests"
OUTPUT_ROOT = ROOT / "perturbed_data"
OUTPUT_TESTS = OUTPUT_ROOT / "tests"
OUTPUT_QUESTIONS = OUTPUT_ROOT / "usaco_questions.json"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def load_questions() -> Dict[str, Any]:
    with QUESTIONS_PATH.open("r") as f:
        return json.load(f)


def choose_problem(data: Dict[str, Any], level: str | None, explicit: str | None) -> str:
    if explicit:
        if explicit not in data:
            raise ValueError(f"Problem id {explicit} not found.")
        return explicit
    ids = list(data.keys())
    if level:
        ids = [pid for pid in ids if data[pid].get("problem_level") == level]
        if not ids:
            raise ValueError(f"No problems found for level={level}")
    return random.choice(ids)


def prompt_perturb(problem_id: str, problem: Dict[str, Any], model: str) -> Dict[str, Any]:
    samples = problem.get("samples") or []
    sample = samples[0] if samples else {"input": "", "output": ""}
    sys_prompt = (
        "You are an expert problem writer. Produce a logically equivalent variant "
        "that keeps the same algorithm/IO contract but changes the story, naming, and surface details. "
        "Do NOT change the underlying logic, constraints, or input/output structure."
    )
    user_prompt = f"""
You are given a competitive programming problem. Create a logically equivalent variant that
keeps the exact input format, output format, constraints, and required algorithm/logic,
but changes the setting/story/variable names/entities/labels. Surface details should differ,
while the computation and correctness conditions stay identical. Be creative with the new setting;
you can substantially re-theme names/entities as long as the logic and IO contract remain intact.

Return JSON with fields:
- new_problem_id: string (default: "{problem_id}_perturb")  # id will be renamed by caller
- perturbed_title: short title (story changed, logic same)
- perturbed_description: full statement (same IO contract & constraints; story/labels changed)
- mapping: bullet points mapping old -> new concepts/labels/entities/variables
- perturbed_sample_input: string (input format identical; values may change if needed to match story)
- perturbed_sample_output: string (matches the perturbed sample input)
- transformer_code: Python code defining two functions:
    def transform_input(text: str) -> str:
        # convert ORIGINAL test input to PERTURBED test input
    def transform_output(text: str) -> str:
        # convert ORIGINAL expected output to PERTURBED expected output
    If the logic is unchanged and no transformation is needed, return identity.

Original problem id: {problem_id}
Original title: {problem.get("name","")}
Original description:
{problem.get("description_no_samples") or problem.get("description") or ""}

Original sample input:
{sample.get("input","")}

Original sample output:
{sample.get("output","")}
"""
    last_err = None
    for attempt in range(5):
        resp = completion(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1800,
        )
        content = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            # remove leading ```json or ``` plus trailing ```
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
            if content.endswith("```"):
                content = content[:-3].strip()
        try:
            start = content.find("{")
            end = content.rfind("}")
            payload = json.loads(content[start : end + 1])
            return payload
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"Failed to parse LLM response as JSON after retries: {last_err}")


def build_transformers(code: str) -> tuple[Callable[[str], str], Callable[[str], str]]:
    # Safe-ish exec in local dict
    local_ns: Dict[str, Any] = {}
    safe_globals = {"__builtins__": {"len": len, "range": range, "print": print, "str": str}}
    try:
        exec(code, safe_globals, local_ns)
    except Exception as e:
        raise RuntimeError(f"Failed to exec transformer_code: {e}")
    transform_input = local_ns.get("transform_input", lambda x: x)
    transform_output = local_ns.get("transform_output", lambda x: x)
    return transform_input, transform_output


def apply_to_tests(
    problem_id: str,
    new_problem_id: str,
    transform_input: Callable[[str], str],
    transform_output: Callable[[str], str],
) -> None:
    src_dir = TESTS_ROOT / problem_id
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Tests not found: {src_dir}")
    dst_dir = OUTPUT_TESTS / new_problem_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    entries = sorted(
        f for f in os.listdir(src_dir)
        if not f.startswith(".") and (src_dir / f).is_file()
    )
    pairs = []
    for fname in entries:
        if fname.endswith(".in"):
            base = fname[:-3]
            out_name = base + ".out"
            if (src_dir / out_name).exists():
                pairs.append((fname, out_name))
        if fname.startswith("I."):
            suffix = fname[2:]
            out_name = "O." + suffix
            if (src_dir / out_name).exists():
                pairs.append((fname, out_name))

    for in_name, out_name in pairs:
        in_text = (src_dir / in_name).read_text()
        out_text = (src_dir / out_name).read_text()
        new_in = transform_input(in_text)
        new_out = transform_output(out_text)
        (dst_dir / in_name).write_text(new_in)
        (dst_dir / out_name).write_text(new_out)


def save_metadata(new_problem_id: str, payload: Dict[str, Any], original: Dict[str, Any]) -> None:
    OUTPUT_ROOT.mkdir(exist_ok=True)
    # Merge into perturbed_data/usaco_questions.json (dict keyed by new_problem_id)
    existing = {}
    if OUTPUT_QUESTIONS.exists():
        try:
            existing = json.loads(OUTPUT_QUESTIONS.read_text())
        except Exception:
            existing = {}

    entry = dict(original)
    entry.update({
        "problem_id": new_problem_id,
        # Keep the original title/name per requirement
        "name": original.get("name"),
        "description": payload.get("perturbed_description", entry.get("description")),
        "description_no_samples": payload.get("perturbed_description", entry.get("description_no_samples")),
        "samples": [{
            "input": payload.get("perturbed_sample_input", ""),
            "output": payload.get("perturbed_sample_output", ""),
        }],
        "mapping": payload.get("mapping", ""),
        "transformer_code": payload.get("transformer_code", ""),
    })

    existing[new_problem_id] = entry
    OUTPUT_QUESTIONS.write_text(json.dumps(existing, indent=2))


def run_once(problem_id: str, problem: Dict[str, Any], run_idx: int, model: str):
    payload = prompt_perturb(problem_id, problem, model)
    # Naming: insert _<run_idx> after the leading question number (e.g., 97 -> 97_1_silver_...)
    def make_new_id(pid: str, idx: int) -> str:
        parts = pid.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            return f"{parts[0]}_{idx}_{parts[1]}"
        return f"{pid}_{idx}"

    new_problem_id = make_new_id(problem_id, run_idx)
    transform_code = payload.get("transformer_code", "")
    transform_input, transform_output = build_transformers(transform_code)
    apply_to_tests(problem_id, new_problem_id, transform_input, transform_output)
    save_metadata(new_problem_id, payload, problem)
    print(f"Saved perturbed problem to {OUTPUT_TESTS / new_problem_id}")


def main():
    parser = argparse.ArgumentParser(description="Perturb a USACO problem and its tests.")
    parser.add_argument("--problem-id", help="Explicit problem id to perturb")
    parser.add_argument("--level", help="Filter by problem_level (e.g., bronze/silver/gold/platinum)")
    parser.add_argument("--runs", type=int, default=3, help="How many perturbations to generate")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model (default: gpt-4o-mini)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set. Please add it to .env or env.local.")

    data = load_questions()
    problem_id = choose_problem(data, args.level, args.problem_id)
    problem = data[problem_id]

    print(f"Selected problem: {problem_id} (level={problem.get('problem_level')})")

    for i in range(1, args.runs + 1):
        run_once(problem_id, problem, i, args.model)


if __name__ == "__main__":
    main()

