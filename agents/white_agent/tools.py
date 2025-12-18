"""
Competitive Programming White Agent
Solves USACO and competitive programming problems using LLM.

Key features:
- Receives problems dynamically from green agent
- Uses GPT-4o/Claude for solution generation
- Optimized for competitive programming problems
- Reads from stdin, writes to stdout
"""

from pathlib import Path
import os
import dotenv
from litellm import completion

# Allow running without agentbeats installed (useful for local testing)
try:
    import agentbeats as ab
except Exception:  # pragma: no cover - simple shim for local runs
    class _Shim:
        def tool(self, fn=None, **kwargs):
            if fn:
                return fn

            def deco(f):
                return f

            return deco

    ab = _Shim()

# Load environment variables explicitly from repo root to avoid CWD surprises
# agents/white_agent/tools.py -> parents[2] = project root
_ROOT = Path(__file__).resolve().parents[2]
dotenv.load_dotenv(_ROOT / ".env")
dotenv.load_dotenv(_ROOT / "env.local")

# Global state to store current problem and conversation
class AgentState:
    def __init__(self):
        self.current_problem = None
        self.conversation_history = []
        self.sample_input = None
        self.sample_output = None
        self.generated_tests = []  # List of {"input": str, "output": str} dicts

    def reset(self):
        self.current_problem = None
        self.conversation_history = []
        self.sample_input = None
        self.sample_output = None
        self.generated_tests = []

# Create global state instance
state = AgentState()

# Flag to enable explanation mode (set EXPLAIN_MODE=true in .env or environment)
EXPLAIN_MODE = os.getenv("EXPLAIN_MODE", "false").lower() in ("true", "1", "yes")

# Flag to enable self-testing mode (set SELF_TEST_MODE=true in .env or environment)
SELF_TEST_MODE = os.getenv("SELF_TEST_MODE", "false").lower() in ("true", "1", "yes")

# Flag to enable generated tests mode (set GENERATE_TESTS_MODE=true in .env or environment)
GENERATE_TESTS_MODE = os.getenv("GENERATE_TESTS_MODE", "false").lower() in ("true", "1", "yes")

# Flag to enable self-review mode (set SELF_REVIEW_MODE=true in .env or environment)
SELF_REVIEW_MODE = os.getenv("SELF_REVIEW_MODE", "false").lower() in ("true", "1", "yes")

# Maximum attempts for self-testing fixes
MAX_FIX_ATTEMPTS = int(os.getenv("MAX_FIX_ATTEMPTS", "3"))

# Number of test cases to generate
NUM_GENERATED_TESTS = int(os.getenv("NUM_GENERATED_TESTS", "3"))

# Maximum self-review rounds
MAX_REVIEW_ROUNDS = int(os.getenv("MAX_REVIEW_ROUNDS", "2"))


def _extract_sample_from_problem(problem_text: str) -> tuple:
    """
    Extract sample input and output from problem statement.

    Args:
        problem_text: The problem statement text

    Returns:
        tuple of (sample_input, sample_output) or (None, None) if not found
    """
    import re

    # Try to find SAMPLE INPUT and SAMPLE OUTPUT sections
    # Pattern 1: "SAMPLE INPUT:\n<input>\nSAMPLE OUTPUT:\n<output>"
    # Pattern 2: "SAMPLE INPUT:\n<input>\nSAMPLE OUTPUT: \n<output>" (with space after colon)

    sample_input = None
    sample_output = None

    # Look for SAMPLE INPUT section
    input_match = re.search(r'SAMPLE INPUT:\s*\n(.*?)(?=\n(?:SAMPLE OUTPUT|OUTPUT FORMAT|$))',
                            problem_text, re.DOTALL | re.IGNORECASE)

    if input_match:
        sample_input = input_match.group(1).strip()

        # Look for SAMPLE OUTPUT section
        output_match = re.search(r'SAMPLE OUTPUT:\s*\n(.*?)(?=\n\n|\n[A-Z]|$)',
                                problem_text, re.DOTALL | re.IGNORECASE)

        if output_match:
            sample_output = output_match.group(1).strip()

    return sample_input, sample_output


@ab.tool
def request_problem_from_green_agent(green_agent_problem: str) -> str:
    """
    Receive and store the competitive programming problem from the green agent.
    Automatically extracts sample input/output from the problem statement.

    Args:
        green_agent_problem: The problem statement from green agent

    Returns:
        Confirmation message
    """
    state.current_problem = green_agent_problem
    state.conversation_history = []  # Reset conversation for new problem

    # Automatically extract sample test from problem statement
    sample_input, sample_output = _extract_sample_from_problem(green_agent_problem)

    if sample_input is not None and sample_output is not None:
        state.sample_input = sample_input
        state.sample_output = sample_output
        return f"Problem received and stored. Sample test extracted automatically."
    else:
        # Clear any old sample data if extraction fails
        state.sample_input = None
        state.sample_output = None
        return "Problem received and stored. Warning: Could not extract sample test from problem statement."


@ab.tool
def receive_sample_test(sample_input: str, expected_output: str) -> str:
    """
    (DEPRECATED - Now optional) Manually set sample test case for self-testing mode.

    Note: As of the latest update, sample tests are automatically extracted from
    the problem statement when request_problem_from_green_agent() is called.
    This tool is kept for backwards compatibility or manual override if needed.

    Args:
        sample_input: The sample input string
        expected_output: The expected output string

    Returns:
        Confirmation message
    """
    state.sample_input = sample_input
    state.sample_output = expected_output
    return "Sample test received (manual override)."


def _test_solution(code: str, test_input: str, expected_output: str) -> dict:
    """
    Test a solution against a test case.

    Args:
        code: The Python code to test
        test_input: Input string for the test
        expected_output: Expected output string

    Returns:
        dict with 'passed', 'actual_output', 'error' keys
    """
    import tempfile
    import subprocess

    try:
        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        # Run the code with the test input
        result = subprocess.run(
            ['python3', temp_file],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=5
        )

        # Clean up
        os.unlink(temp_file)

        actual_output = result.stdout.strip()
        expected = expected_output.strip()

        return {
            'passed': actual_output == expected,
            'actual_output': actual_output,
            'expected_output': expected,
            'error': result.stderr if result.stderr else None
        }

    except subprocess.TimeoutExpired:
        return {
            'passed': False,
            'actual_output': '',
            'expected_output': expected_output,
            'error': 'Solution timed out (exceeded 5 seconds)'
        }
    except Exception as e:
        return {
            'passed': False,
            'actual_output': '',
            'expected_output': expected_output,
            'error': str(e)
        }


def _fix_solution(original_code: str, test_result: dict, problem: str, attempt: int) -> str:
    """
    Ask LLM to fix a solution based on test results.
    LLM provides analysis and explanation, but only code is returned.

    Args:
        original_code: The code that failed
        test_result: Results from _test_solution
        problem: The problem statement
        attempt: Which attempt this is (for limiting retries)

    Returns:
        Fixed code (analysis/explanation printed to stdout)
    """
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        return original_code

    try:
        model = "gpt-4o"
        if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            model = "claude-3-5-sonnet-20241022"

        error_info = ""
        if test_result.get('error'):
            error_info = f"\n\nError encountered:\n{test_result['error']}"

        fix_prompt = f"""Your solution for this problem failed the sample test:

PROBLEM:
{problem}

YOUR CODE:
{original_code}

TEST RESULT:
Expected output: {test_result['expected_output']}
Actual output: {test_result['actual_output']}{error_info}

Please provide a structured response with your analysis and the fix.

Format your response EXACTLY as follows:

ANALYSIS:
[Explain what went wrong in the original code - be specific about the bug or logical error]

CHANGES:
[Describe the specific changes you're making to fix the issue]

FIXED CODE:
[Provide ONLY the corrected Python code here]

CRITICAL for the code:
- Code must EXECUTE IMMEDIATELY when run (no uncalled functions)
- Read from stdin, process, print to stdout
- If you use functions, call them at the end

This is attempt {attempt} of {MAX_FIX_ATTEMPTS}.
"""

        print(f"\n{'='*70}")
        print(f"SELF-TEST MODE: Attempt {attempt} - Fixing solution...")
        print(f"{'='*70}")
        print(f"Expected: {test_result['expected_output']}")
        print(f"Got: {test_result['actual_output']}")
        if test_result.get('error'):
            print(f"Error: {test_result['error']}")
        print(f"{'='*70}\n")

        response = completion(
            model=model,
            messages=[{"role": "user", "content": fix_prompt}],
            temperature=0.0,
            max_tokens=2000
        )

        response_text = response.choices[0].message.content

        # Parse the structured response
        analysis = ""
        changes = ""
        fixed_code = response_text

        if "ANALYSIS:" in response_text and "CHANGES:" in response_text and "FIXED CODE:" in response_text:
            parts = response_text.split("FIXED CODE:")
            before_code = parts[0]
            fixed_code = parts[1].strip()

            # Extract analysis and changes
            if "ANALYSIS:" in before_code:
                analysis_parts = before_code.split("ANALYSIS:")[1]
                if "CHANGES:" in analysis_parts:
                    analysis = analysis_parts.split("CHANGES:")[0].strip()
                    changes = analysis_parts.split("CHANGES:")[1].strip()
                else:
                    analysis = analysis_parts.strip()

            # Print analysis and changes
            if analysis:
                print("ANALYSIS:")
                print("-" * 70)
                print(analysis)
                print("-" * 70)
                print()

            if changes:
                print("CHANGES:")
                print("-" * 70)
                print(changes)
                print("-" * 70)
                print()
        else:
            # Fallback: treat entire response as code
            print("Warning: LLM did not follow structured format")
            print()

        # Clean markdown from code
        if "```python" in fixed_code:
            fixed_code = fixed_code.split("```python")[1].split("```")[0].strip()
        elif "```" in fixed_code:
            fixed_code = fixed_code.split("```")[1].split("```")[0].strip()

        print("FIXED CODE:")
        print("-" * 70)
        print(fixed_code[:200] + ("..." if len(fixed_code) > 200 else ""))
        print("-" * 70)
        print()

        return fixed_code

    except Exception as e:
        print(f"Error during fix attempt: {e}")
        return original_code


def _generate_test_cases(problem: str, num_tests: int = 3) -> list:
    """
    Ask LLM to generate test cases based on problem understanding.

    Args:
        problem: The problem statement
        num_tests: Number of test cases to generate

    Returns:
        List of test case dicts with 'input' and 'expected_output' keys
    """
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        return []

    try:
        model = "gpt-4o"
        if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            model = "claude-3-5-sonnet-20241022"

        print(f"\n{'='*70}")
        print(f"GENERATE_TESTS_MODE: Generating {num_tests} test cases...")
        print(f"{'='*70}\n")

        prompt = f"""Based on this competitive programming problem, generate {num_tests} diverse test cases.

PROBLEM:
{problem}

Generate test cases that cover:
1. The sample case (if given in problem)
2. Edge cases (minimum/maximum values, boundary conditions)
3. Typical cases

For EACH test case, provide:
1. A brief description of what it tests
2. The input
3. The expected output

Format your response EXACTLY as follows:

TEST 1:
DESCRIPTION: [What this test case covers]
INPUT:
[The exact input text]
EXPECTED OUTPUT:
[The exact expected output]

TEST 2:
DESCRIPTION: [What this test case covers]
INPUT:
[The exact input text]
EXPECTED OUTPUT:
[The exact expected output]

... continue for all {num_tests} tests

IMPORTANT:
- Input and output should be EXACT strings that would be used for stdin/stdout
- No extra explanations or markdown in the INPUT/EXPECTED OUTPUT sections
- Make sure outputs are correct based on the problem logic
"""

        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Slightly creative for diverse tests
            max_tokens=2000
        )

        response_text = response.choices[0].message.content

        # Parse the test cases
        test_cases = []
        import re

        # Find all TEST blocks
        test_blocks = re.findall(
            r'TEST \d+:.*?DESCRIPTION:(.*?)INPUT:(.*?)EXPECTED OUTPUT:(.*?)(?=TEST \d+:|$)',
            response_text,
            re.DOTALL
        )

        for i, (description, input_text, output_text) in enumerate(test_blocks):
            test_input = input_text.strip()
            test_output = output_text.strip()
            test_desc = description.strip()

            test_cases.append({
                "input": test_input,
                "expected_output": test_output,
                "description": test_desc
            })

            print(f"Generated Test {i+1}:")
            print(f"  Description: {test_desc}")
            print(f"  Input: {test_input[:50]}{'...' if len(test_input) > 50 else ''}")
            print(f"  Expected: {test_output[:50]}{'...' if len(test_output) > 50 else ''}")
            print()

        print(f"{'='*70}")
        print(f"Generated {len(test_cases)} test cases")
        print(f"{'='*70}\n")

        return test_cases

    except Exception as e:
        print(f"Error generating test cases: {e}")
        return []


def _revise_code_or_tests(code: str, tests: list, failed_test_idx: int,
                          test_result: dict, problem: str, attempt: int) -> dict:
    """
    Ask LLM whether to revise the code or revise the test cases.

    Args:
        code: The current code
        tests: List of current test cases
        failed_test_idx: Index of the test that failed
        test_result: Results from _test_solution
        problem: The problem statement
        attempt: Current attempt number

    Returns:
        dict with 'action' ('fix_code' or 'fix_tests'), 'new_code' or 'new_tests', and 'reasoning'
    """
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        return {"action": "fix_code", "new_code": code, "reasoning": "LLM unavailable"}

    try:
        model = "gpt-4o"
        if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            model = "claude-3-5-sonnet-20241022"

        failed_test = tests[failed_test_idx]

        prompt = f"""Your solution failed a test case you generated. Analyze whether the CODE is wrong or the TEST is wrong.

PROBLEM:
{problem}

YOUR CODE:
{code}

FAILED TEST (Test #{failed_test_idx + 1}):
Description: {failed_test.get('description', 'N/A')}
Input: {failed_test['input']}
Expected Output: {failed_test['expected_output']}
Actual Output: {test_result['actual_output']}

Error: {test_result.get('error', 'None')}

Analyze:
1. Is the code logic correct for this problem?
2. Is the test case valid and correctly computed?
3. Should we fix the code or fix the test?

Provide your response in this format:

DECISION: [Either "FIX_CODE" or "FIX_TESTS"]

REASONING:
[Explain your analysis - what is wrong and why]

[If DECISION is FIX_CODE:]
FIXED CODE:
[Corrected code]

[If DECISION is FIX_TESTS:]
CORRECTED TEST:
INPUT:
[Corrected input]
EXPECTED OUTPUT:
[Corrected expected output]
DESCRIPTION:
[Updated description]

Be thorough in your reasoning. This is attempt {attempt} of {MAX_FIX_ATTEMPTS}.
"""

        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000
        )

        response_text = response.choices[0].message.content

        # Parse the decision
        decision = "fix_code"  # Default
        if "DECISION:" in response_text:
            decision_line = response_text.split("DECISION:")[1].split("\n")[0].strip()
            if "FIX_TESTS" in decision_line.upper():
                decision = "fix_tests"
            elif "FIX_CODE" in decision_line.upper():
                decision = "fix_code"

        # Extract reasoning
        reasoning = ""
        if "REASONING:" in response_text:
            reasoning_part = response_text.split("REASONING:")[1]
            if decision == "fix_code" and "FIXED CODE:" in reasoning_part:
                reasoning = reasoning_part.split("FIXED CODE:")[0].strip()
            elif decision == "fix_tests" and "CORRECTED TEST:" in reasoning_part:
                reasoning = reasoning_part.split("CORRECTED TEST:")[0].strip()
            else:
                reasoning = reasoning_part.strip()

        result = {
            "action": decision,
            "reasoning": reasoning
        }

        if decision == "fix_code":
            # Extract fixed code
            if "FIXED CODE:" in response_text:
                fixed_code = response_text.split("FIXED CODE:")[1].strip()
                # Clean markdown
                if "```python" in fixed_code:
                    fixed_code = fixed_code.split("```python")[1].split("```")[0].strip()
                elif "```" in fixed_code:
                    fixed_code = fixed_code.split("```")[1].split("```")[0].strip()
                result["new_code"] = fixed_code
            else:
                result["new_code"] = code

        elif decision == "fix_tests":
            # Extract corrected test
            new_tests = tests.copy()
            corrected_test = {}

            if "INPUT:" in response_text and "EXPECTED OUTPUT:" in response_text:
                import re
                input_match = re.search(r'INPUT:\s*\n(.*?)(?=EXPECTED OUTPUT:)', response_text, re.DOTALL)
                output_match = re.search(r'EXPECTED OUTPUT:\s*\n(.*?)(?=DESCRIPTION:|$)', response_text, re.DOTALL)
                desc_match = re.search(r'DESCRIPTION:\s*\n(.*?)$', response_text, re.DOTALL)

                if input_match:
                    corrected_test["input"] = input_match.group(1).strip()
                if output_match:
                    corrected_test["expected_output"] = output_match.group(1).strip()
                if desc_match:
                    corrected_test["description"] = desc_match.group(1).strip()
                else:
                    corrected_test["description"] = failed_test.get("description", "")

                # Update the test in the list
                new_tests[failed_test_idx] = corrected_test
                result["new_tests"] = new_tests
            else:
                result["new_tests"] = tests

        return result

    except Exception as e:
        print(f"Error during revision analysis: {e}")
        return {"action": "fix_code", "new_code": code, "reasoning": f"Error: {e}"}


def _test_with_generated_tests(initial_solution: str) -> str:
    """
    Test solution against generated test cases and iteratively fix code or tests.

    Args:
        initial_solution: The initially generated solution

    Returns:
        Final solution after testing and fixes
    """
    # Generate test cases first
    test_cases = _generate_test_cases(state.current_problem, NUM_GENERATED_TESTS)

    if not test_cases:
        print("Warning: Could not generate test cases. Skipping generated tests mode.")
        return initial_solution

    # Store in state for reference
    state.generated_tests = test_cases

    print(f"\n{'='*70}")
    print("GENERATE_TESTS_MODE: Testing solution against generated tests...")
    print(f"{'='*70}\n")

    current_solution = initial_solution
    current_tests = test_cases.copy()

    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        all_passed = True
        failed_test_idx = None
        failed_result = None

        # Test against all generated tests
        for i, test in enumerate(current_tests):
            test_result = _test_solution(
                current_solution,
                test["input"],
                test["expected_output"]
            )

            if not test_result['passed']:
                all_passed = False
                failed_test_idx = i
                failed_result = test_result
                print(f"Test {i+1}/{len(current_tests)} FAILED")
                print(f"  Description: {test.get('description', 'N/A')}")
                print(f"  Expected: {test_result['expected_output']}")
                print(f"  Got: {test_result['actual_output']}")
                if test_result.get('error'):
                    print(f"  Error: {test_result['error']}")
                print()
                break
            else:
                print(f"Test {i+1}/{len(current_tests)} PASSED")

        if all_passed:
            print(f"\n{'='*70}")
            print(f"✓ ALL GENERATED TESTS PASSED on attempt {attempt}!")
            print(f"{'='*70}\n")
            return current_solution

        # If we have attempts left, try to fix
        if attempt < MAX_FIX_ATTEMPTS:
            print(f"\n{'='*70}")
            print(f"GENERATE_TESTS_MODE: Attempt {attempt + 1} - Analyzing failure...")
            print(f"{'='*70}\n")

            revision = _revise_code_or_tests(
                current_solution,
                current_tests,
                failed_test_idx,
                failed_result,
                state.current_problem,
                attempt + 1
            )

            print(f"DECISION: {revision['action'].upper()}")
            print(f"\nREASONING:")
            print("-" * 70)
            print(revision['reasoning'])
            print("-" * 70)
            print()

            if revision['action'] == 'fix_code':
                current_solution = revision.get('new_code', current_solution)
                print("Updated code. Retesting...\n")
            elif revision['action'] == 'fix_tests':
                current_tests = revision.get('new_tests', current_tests)
                state.generated_tests = current_tests  # Update state
                print("Updated test case. Retesting...\n")
        else:
            # Final attempt failed
            print(f"\n{'='*70}")
            print(f"✗ GENERATED TESTS FAILED after {MAX_FIX_ATTEMPTS} attempts")
            print(f"Returning last version anyway...")
            print(f"{'='*70}\n")
            return current_solution

    return current_solution


def _self_review_solution(initial_solution: str) -> str:
    """
    LLM reviews its own solution and iteratively improves it.

    Args:
        initial_solution: The initially generated solution

    Returns:
        Final solution after self-review and improvements
    """
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        return initial_solution

    print(f"\n{'='*70}")
    print("SELF-REVIEW MODE: Starting code review process...")
    print(f"{'='*70}\n")

    current_solution = initial_solution

    for round_num in range(1, MAX_REVIEW_ROUNDS + 1):
        try:
            model = "gpt-4o"
            if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
                model = "claude-3-5-sonnet-20241022"

            prompt = f"""You are a code reviewer. Critically analyze this solution for a competitive programming problem.

PROBLEM:
{state.current_problem}

CURRENT SOLUTION (Review Round {round_num}):
{current_solution}

Perform a thorough code review:
1. Check for logical errors or bugs
2. Verify correctness of algorithm
3. Check edge case handling
4. Look for off-by-one errors
5. Verify input/output handling
6. Check for runtime issues (infinite loops, etc.)
7. Verify the code matches problem requirements

Provide your review in this format:

REVIEW ANALYSIS:
[Your thorough analysis of the code]

ISSUES FOUND:
[List each potential issue, or write "None" if code looks correct]
1. [Issue 1]
2. [Issue 2]
...

SEVERITY: [CRITICAL / MINOR / NONE]
- CRITICAL: Code has bugs that will cause wrong answers
- MINOR: Code works but could be improved
- NONE: Code is correct, no changes needed

DECISION: [FIX_CODE or KEEP_AS_IS]

[If DECISION is FIX_CODE:]
CHANGES:
[Describe what you're changing and why]

FIXED CODE:
[Provide the corrected code]

Be very critical and thorough. This is review round {round_num} of {MAX_REVIEW_ROUNDS}.
"""

            print(f"{'='*70}")
            print(f"SELF-REVIEW MODE: Review Round {round_num}/{MAX_REVIEW_ROUNDS}")
            print(f"{'='*70}\n")

            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2500
            )

            response_text = response.choices[0].message.content

            # Parse the response
            review_analysis = ""
            issues_found = ""
            severity = "NONE"
            decision = "KEEP_AS_IS"
            changes = ""
            fixed_code = current_solution

            # Extract review analysis
            if "REVIEW ANALYSIS:" in response_text:
                analysis_part = response_text.split("REVIEW ANALYSIS:")[1]
                if "ISSUES FOUND:" in analysis_part:
                    review_analysis = analysis_part.split("ISSUES FOUND:")[0].strip()
                else:
                    review_analysis = analysis_part.strip()

            # Extract issues found
            if "ISSUES FOUND:" in response_text:
                issues_part = response_text.split("ISSUES FOUND:")[1]
                if "SEVERITY:" in issues_part:
                    issues_found = issues_part.split("SEVERITY:")[0].strip()
                elif "DECISION:" in issues_part:
                    issues_found = issues_part.split("DECISION:")[0].strip()
                else:
                    issues_found = issues_part.strip()

            # Extract severity
            if "SEVERITY:" in response_text:
                severity_line = response_text.split("SEVERITY:")[1].split("\n")[0].strip()
                if "CRITICAL" in severity_line.upper():
                    severity = "CRITICAL"
                elif "MINOR" in severity_line.upper():
                    severity = "MINOR"
                else:
                    severity = "NONE"

            # Extract decision
            if "DECISION:" in response_text:
                decision_line = response_text.split("DECISION:")[1].split("\n")[0].strip()
                if "FIX_CODE" in decision_line.upper():
                    decision = "FIX_CODE"
                elif "KEEP" in decision_line.upper():
                    decision = "KEEP_AS_IS"

            # Print review analysis
            print("REVIEW ANALYSIS:")
            print("-" * 70)
            print(review_analysis if review_analysis else "No analysis provided")
            print("-" * 70)
            print()

            print("ISSUES FOUND:")
            print("-" * 70)
            print(issues_found if issues_found else "None")
            print("-" * 70)
            print()

            print(f"SEVERITY: {severity}")
            print(f"DECISION: {decision}")
            print()

            # If decision is to keep as-is, we're done
            if decision == "KEEP_AS_IS":
                print(f"{'='*70}")
                print(f"✓ SELF-REVIEW COMPLETE: Code approved after {round_num} round(s)")
                print(f"{'='*70}\n")
                return current_solution

            # Otherwise, extract changes and fixed code
            if decision == "FIX_CODE":
                # Extract changes
                if "CHANGES:" in response_text:
                    changes_part = response_text.split("CHANGES:")[1]
                    if "FIXED CODE:" in changes_part:
                        changes = changes_part.split("FIXED CODE:")[0].strip()
                    else:
                        changes = changes_part.strip()

                # Extract fixed code
                if "FIXED CODE:" in response_text:
                    fixed_code = response_text.split("FIXED CODE:")[1].strip()
                    # Clean markdown
                    if "```python" in fixed_code:
                        fixed_code = fixed_code.split("```python")[1].split("```")[0].strip()
                    elif "```" in fixed_code:
                        fixed_code = fixed_code.split("```")[1].split("```")[0].strip()

                print("CHANGES:")
                print("-" * 70)
                print(changes if changes else "No changes description provided")
                print("-" * 70)
                print()

                print("FIXED CODE:")
                print("-" * 70)
                print(fixed_code[:300] + ("..." if len(fixed_code) > 300 else ""))
                print("-" * 70)
                print()

                current_solution = fixed_code
                print(f"Code updated. {'Starting next review round...' if round_num < MAX_REVIEW_ROUNDS else 'Final version.'}\n")

        except Exception as e:
            print(f"Error during review round {round_num}: {e}")
            print("Keeping current version.\n")
            break

    # If we've gone through all rounds and still making changes
    print(f"{'='*70}")
    print(f"SELF-REVIEW COMPLETE: Completed {MAX_REVIEW_ROUNDS} review round(s)")
    print(f"{'='*70}\n")

    return current_solution


@ab.tool
def get_current_problem() -> str:
    """
    Get the currently stored problem.
    Returns the problem that was set by the green agent.
    """
    if state.current_problem is None:
        return "No problem has been set yet. Green agent should provide a problem first."
    return state.current_problem


def _solve_with_explanations() -> str:
    """
    Generate solution with plan and explanations (for EXPLAIN_MODE).
    Prints explanations to stdout but returns only executable code.
    """
    try:
        model = "gpt-4o"
        if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            model = "claude-3-5-sonnet-20241022"

        # Step 1: Generate plan and analysis
        print("\n" + "="*70)
        print("EXPLANATION MODE: Generating plan and analysis...")
        print("="*70 + "\n")

        analysis_response = completion(
            model=model,
            messages=[{
                "role": "user",
                "content": f"""Analyze this competitive programming problem and create a solution plan:

{state.current_problem}

Provide:
1. **Problem Understanding**: What is the problem asking for?
2. **Input/Output Format**: How should we read input and format output?
3. **Approach**: What algorithm or strategy will solve this?
4. **Key Insights**: Any edge cases or important observations?
5. **Step-by-Step Plan**: Outline the solution steps

Be concise but thorough.
"""
            }],
            temperature=0.0
        )

        plan = analysis_response.choices[0].message.content

        # Print the plan for visibility
        print("SOLUTION PLAN:")
        print("-"*70)
        print(plan)
        print("-"*70 + "\n")

        # Step 2: Generate code with explanations
        print("Generating code with explanations...\n")

        code_response = completion(
            model=model,
            messages=[
                {"role": "assistant", "content": plan},
                {"role": "user", "content": """Now provide the solution in this format:

1. First, explain your implementation approach in 2-3 sentences
2. Then provide the complete executable Python code

CRITICAL for the code:
- Code must EXECUTE IMMEDIATELY when run (no uncalled functions)
- Read from stdin, process, print to stdout
- If you use functions, call them at the end
- Provide ONLY executable code in the code section

Format:
EXPLANATION:
[Your 2-3 sentence explanation]

CODE:
[Executable Python code only]
"""}
            ],
            temperature=0.0
        )

        response_text = code_response.choices[0].message.content

        # Parse explanation and code
        if "CODE:" in response_text:
            parts = response_text.split("CODE:")
            explanation = parts[0].replace("EXPLANATION:", "").strip()
            code = parts[1].strip()
        else:
            explanation = "No explanation provided"
            code = response_text

        # Print explanation
        print("IMPLEMENTATION EXPLANATION:")
        print("-"*70)
        print(explanation)
        print("-"*70 + "\n")

        # Clean markdown from code if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        print("CODE GENERATED:")
        print("-"*70)
        print(code[:200] + ("..." if len(code) > 200 else ""))
        print("-"*70 + "\n")
        print("="*70)
        print("END OF EXPLANATION MODE")
        print("="*70 + "\n")

        # Return only the executable code (not the explanations)
        return code

    except Exception as e:
        print(f"LLM error in explanation mode: {e}")
        return get_fallback_solution()


def _self_test_and_fix(initial_solution: str) -> str:
    """
    Test solution against sample and fix if it fails.

    Args:
        initial_solution: The initially generated solution

    Returns:
        Final solution (fixed if necessary)
    """
    print(f"\n{'='*70}")
    print("SELF-TEST MODE: Testing solution against sample...")
    print(f"{'='*70}\n")

    current_solution = initial_solution

    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        # Test the solution
        test_result = _test_solution(
            current_solution,
            state.sample_input,
            state.sample_output
        )

        if test_result['passed']:
            print(f"{'='*70}")
            print(f"✓ SELF-TEST PASSED on attempt {attempt}!")
            print(f"{'='*70}\n")
            return current_solution

        # If failed and we have attempts left, try to fix
        if attempt < MAX_FIX_ATTEMPTS:
            print(f"✗ Test failed on attempt {attempt}")
            current_solution = _fix_solution(
                current_solution,
                test_result,
                state.current_problem,
                attempt + 1
            )
        else:
            # Final attempt failed
            print(f"\n{'='*70}")
            print(f"✗ SELF-TEST FAILED after {MAX_FIX_ATTEMPTS} attempts")
            print(f"Expected: {test_result['expected_output']}")
            print(f"Got: {test_result['actual_output']}")
            if test_result.get('error'):
                print(f"Error: {test_result['error']}")
            print(f"Returning last attempt anyway...")
            print(f"{'='*70}\n")
            return current_solution

    return current_solution


@ab.tool
def solve_problem() -> str:
    """
    Generate a solution for the competitive programming problem using LLM.
    If EXPLAIN_MODE is enabled, generates plan and explanations but still returns only code.
    If SELF_REVIEW_MODE is enabled, LLM reviews and improves its own code.
    If SELF_TEST_MODE is enabled, tests and fixes solution against sample test.
    If GENERATE_TESTS_MODE is enabled, generates tests and iteratively fixes code or tests.
    """
    if state.current_problem is None:
        return "Error: No problem to solve. Please set a problem first."

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        return get_fallback_solution()

    # Generate initial solution (with or without explanations)
    if EXPLAIN_MODE:
        solution = _solve_with_explanations()
    else:
        solution = _generate_solution_standard()

    # If self-review mode is enabled, review and improve the code
    if SELF_REVIEW_MODE:
        solution = _self_review_solution(solution)

    # If self-test mode is enabled and we have sample data, test and fix if needed
    if SELF_TEST_MODE and state.sample_input is not None and state.sample_output is not None:
        solution = _self_test_and_fix(solution)

    # If generate tests mode is enabled, generate tests and validate/fix
    if GENERATE_TESTS_MODE:
        solution = _test_with_generated_tests(solution)

    return solution


def _generate_solution_standard() -> str:
    """Generate solution using standard (non-explanation) approach."""
    try:
        system_prompt = """You are an expert competitive programmer specializing in algorithms and data structures.
Generate complete, executable Python code that runs immediately and produces output.
CRITICAL: Do not wrap code in function definitions unless you call them at the end.
The code must execute directly when run - it should read stdin, process, and print output."""

        user_prompt = f"""Solve this competitive programming problem:

{state.current_problem}

Requirements:
1. Read input from stdin using sys.stdin.read()
2. Parse and process all test cases as specified
3. Print results to stdout in the exact required format
4. Code must EXECUTE IMMEDIATELY when run (no uncalled functions)
5. Use efficient algorithms and handle edge cases

IMPORTANT:
- If you define functions, you MUST call them (e.g., if you write "def solve():", end with "solve()")
- Better yet, write direct executable code without function wrappers
- The code should produce output when run, not just define functions

Provide ONLY the Python code, no explanations, no markdown formatting.
"""

        # Choose model based on API key availability
        model = "gpt-4o"
        if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            model = "claude-3-5-sonnet-20241022"

        print(f"Generating solution using {model}...")

        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=2000
        )

        solution = response.choices[0].message.content

        # Clean markdown formatting if present
        if "```python" in solution:
            solution = solution.split("```python")[1].split("```")[0].strip()
        elif "```" in solution:
            solution = solution.split("```")[1].split("```")[0].strip()

        print("Solution generated successfully!")
        return solution

    except Exception as e:
        print(f"LLM error: {e}")
        return get_fallback_solution()


@ab.tool
def solve_with_reasoning() -> dict:
    """
    Generate solution with step-by-step reasoning.
    Returns both the reasoning and the solution.
    """
    if state.current_problem is None:
        return {"error": "No problem to solve"}

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        return {
            "reasoning": "LLM not available",
            "solution": get_fallback_solution()
        }

    try:
        model = "gpt-4o"
        if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            model = "claude-3-5-sonnet-20241022"

        # Step 1: Analyze the problem
        analysis_response = completion(
            model=model,
            messages=[{
                "role": "user",
                "content": f"""Analyze this problem step by step:

{state.current_problem}

Provide:
1. Problem understanding
2. Key constraints and requirements
3. Solution approach
4. Algorithm/strategy to use
5. Edge cases to consider
"""
            }],
            temperature=0.0
        )

        reasoning = analysis_response.choices[0].message.content

        # Step 2: Generate solution based on analysis
        code_response = completion(
            model=model,
            messages=[
                {"role": "assistant", "content": reasoning},
                {"role": "user", "content": """Now provide the complete solution code based on this analysis.

CRITICAL: The code must EXECUTE IMMEDIATELY when run.
- If you use functions, call them at the end
- Better yet, write direct executable code
- Code should read stdin, process, and print output
- No uncalled function definitions

Provide ONLY executable Python code."""}
            ],
            temperature=0.0
        )

        solution = code_response.choices[0].message.content

        # Clean markdown
        if "```python" in solution:
            solution = solution.split("```python")[1].split("```")[0].strip()
        elif "```" in solution:
            solution = solution.split("```")[1].split("```")[0].strip()

        return {
            "reasoning": reasoning,
            "solution": solution
        }

    except Exception as e:
        print(f"LLM error: {e}")
        return {
            "error": str(e),
            "solution": get_fallback_solution()
        }


@ab.tool
def get_my_solution() -> str:
    """
    Main tool called by evaluation systems.
    Generates solution for the current problem.
    """
    return solve_problem()


@ab.tool
def get_fallback_solution() -> str:
    """
    Fallback when LLM is unavailable.
    Attempts basic heuristics or returns error message.
    """
    if state.current_problem is None:
        return '''import sys
print("Error: No problem provided")
'''

    # Try to provide a safe fallback
    return '''import sys
# LLM unavailable - basic fallback solution
input_data = sys.stdin.read().strip()
print("Unable to generate solution - LLM not configured")
'''


@ab.tool
def chat_about_problem(user_message: str) -> str:
    """
    Interactive chat about the problem.
    Allows multi-turn conversation for problem clarification.

    Args:
        user_message: Question or comment about the problem

    Returns:
        AI response
    """
    if state.current_problem is None:
        return "No problem is currently set. Please provide a problem first."

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        return "LLM not available for chat."

    try:
        # Add user message to history
        state.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Build messages with problem context
        messages = [
            {
                "role": "system",
                "content": f"You are helping solve this problem:\n\n{state.current_problem}\n\nAnswer questions and provide guidance."
            }
        ] + state.conversation_history

        model = "gpt-4o-mini"  # Use cheaper model for chat
        if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            model = "claude-3-5-haiku-20241022"

        response = completion(
            model=model,
            messages=messages,
            temperature=0.7  # More creative for chat
        )

        assistant_message = response.choices[0].message.content

        # Add to history
        state.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

    except Exception as e:
        return f"Chat error: {e}"


@ab.tool
def reset_agent() -> str:
    """
    Reset agent state for a new problem.
    Clears current problem and conversation history.
    """
    state.reset()
    return "Agent state reset. Ready for new problem."


@ab.tool
def get_agent_status() -> dict:
    """
    Get current agent status and configuration.
    Useful for debugging and monitoring.
    """
    return {
        "has_problem": state.current_problem is not None,
        "conversation_turns": len(state.conversation_history) // 2,
        "llm_available": bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")),
        "available_models": {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "google": bool(os.getenv("GOOGLE_API_KEY"))
        }
    }


@ab.tool
def get_buggy_solution() -> str:
    """
    Intentionally buggy solution for testing evaluators.
    """
    return '''import sys
# Intentionally incorrect solution for testing
lines = sys.stdin.read().strip().split("\\n")
T = int(lines[0])
for _ in range(T):
    print("0")  # Always wrong
'''


# Example usage
if __name__ == "__main__":
    import sys

    print("Testing General LLM-Powered White Agent")
    print("=" * 50)

    # Simulate receiving a problem from green agent
    test_problem = """
Calculate the sum of all prime numbers less than 100.

Input: None (no input required)
Output: A single integer - the sum of all primes less than 100
"""

    print("\n1. Receiving problem from green agent...")
    result = request_problem_from_green_agent(test_problem)
    print(f"   {result}")

    print("\n2. Checking current problem...")
    current = get_current_problem()
    print(f"   {current[:100]}...")

    print("\n3. Checking agent status...")
    status = get_agent_status()
    print(f"   Status: {status}")

    print("\n4. Generating solution...")
    if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        solution = solve_problem()
        print(f"   Solution generated: {len(solution)} characters")
        print(f"   Preview: {solution[:200]}...")
    else:
        print("   ⚠️  No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env")

    print("\n5. Testing chat...")
    chat_response = chat_about_problem("What algorithm should I use?")
    print(f"   Chat response: {chat_response[:150]}...")

    print("\n" + "=" * 50)
    print("Test complete!")
