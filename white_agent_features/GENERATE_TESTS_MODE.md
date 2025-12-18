# White Agent Generate Tests Mode

The white agent supports a **Generate Tests Mode** where an LLM generates test cases based on its understanding of the problem, tests the solution against them, and can iteratively revise either the code or the tests when failures occur.

## How It Works

When `GENERATE_TESTS_MODE` is enabled, the white agent:

1. **Generates Initial Solution**
   - Uses standard or explanation mode (depending on EXPLAIN_MODE flag)
   - May first test against sample if SELF_TEST_MODE is also enabled

2. **Generates Test Cases**
   - LLM analyzes the problem statement
   - Creates diverse test cases covering:
     - Sample case (from problem statement)
     - Edge cases (min/max values, boundaries)
     - Typical cases
   - Default: 3 test cases (configurable via `NUM_GENERATED_TESTS`)

3. **Tests Solution**
   - Runs the solution against each generated test
   - Captures actual vs expected output
   - Stops at first failure

4. **Analyzes Failures**
   - LLM determines whether the CODE is wrong or the TEST is wrong
   - Provides detailed reasoning for its decision

5. **Fixes Code or Tests**
   - If code is wrong: Generates fixed code
   - If test is wrong: Generates corrected test case
   - Maximum attempts: `MAX_FIX_ATTEMPTS` (default: 3)

6. **Returns Final Solution**
   - Always returns only executable code to green agent
   - Test generation, results, and revisions printed to stdout

## Why This Mode is Useful

### Advantages over SELF_TEST_MODE

1. **More Comprehensive Testing**
   - Tests multiple cases, not just the sample
   - Covers edge cases that sample might miss

2. **Self-Correcting Test Cases**
   - If LLM misunderstands the problem, it can fix its tests
   - Prevents false negatives from incorrect expected outputs

3. **Better Problem Understanding**
   - Generating tests forces deeper problem analysis
   - Can catch subtle requirements missed in first read

4. **Iterative Refinement**
   - Code and tests evolve together
   - More robust final solution


## Enabling Generate Tests Mode

### Option 1: Environment Variable (Temporary)
```bash
GENERATE_TESTS_MODE=true bash run_test_agents_directly.sh
```

### Option 2: Update .env File (Permanent)
Edit `agent_ai/Newer stuff/usaco/white_agent/.env`:
```bash
# Enable the mode
GENERATE_TESTS_MODE=true

# Optionally adjust number of tests and max attempts
NUM_GENERATED_TESTS=5
MAX_FIX_ATTEMPTS=5
```

### Option 3: Combine with Other Modes
```bash
# With explanation mode
EXPLAIN_MODE=true GENERATE_TESTS_MODE=true bash run_test_agents_directly.sh

# With self-test mode (tests sample first, then generated tests)
SELF_TEST_MODE=true GENERATE_TESTS_MODE=true bash run_test_agents_directly.sh

# All modes
EXPLAIN_MODE=true SELF_TEST_MODE=true GENERATE_TESTS_MODE=true bash run_test_agents_directly.sh
```
