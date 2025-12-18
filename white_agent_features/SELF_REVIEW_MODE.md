# White Agent Self-Review Mode

The white agent supports a **Self-Review Mode** where the LLM acts as its own code reviewer, critically analyzing the solution for flaws and iteratively improving it without running any tests.

## How It Works

When `SELF_REVIEW_MODE` is enabled, the white agent:

1. **Generates Initial Solution**
   - Uses standard or explanation mode (depending on EXPLAIN_MODE flag)

2. **Critical Code Review** (Round 1)
   - LLM reviews its own code with a critical eye
   - Checks for:
     - Logical errors or bugs
     - Algorithm correctness
     - Edge case handling
     - Off-by-one errors
     - Input/output handling issues
     - Runtime issues (infinite loops, etc.)
     - Problem requirement compliance

3. **Issues Analysis**
   - Lists all potential problems found
   - Assigns severity: CRITICAL, MINOR, or NONE
   - Makes decision: FIX_CODE or KEEP_AS_IS

4. **Code Improvement** (if needed)
   - Describes specific changes to make
   - Provides fixed code
   - Proceeds to next review round

5. **Iteration**
   - Repeats review process up to `MAX_REVIEW_ROUNDS` times
   - Stops early if code is approved (KEEP_AS_IS decision)

6. **Returns Final Solution**
   - Always returns only executable code to green agent
   - All review analysis and changes printed to stdout

## Why This Mode is Useful

### Advantages

1. **Pre-Test Bug Catching**
   - Finds bugs before running any tests
   - Pure logical analysis, no test cases needed
   - Catches subtle issues tests might miss

2. **Works Without Tests**
   - No sample input/output required
   - Useful when test data is unavailable
   - Complements testing modes

3. **Deep Code Analysis**
   - Reviews algorithm correctness
   - Checks edge case handling
   - Verifies problem understanding
   - Catches off-by-one errors

4. **Iterative Improvement**
   - Multiple review rounds
   - Each round improves the code
   - Converges to cleaner solution

5. **Self-Correction**
   - LLM corrects its own mistakes
   - Learns from its initial implementation
   - No external feedback needed


## Enabling Self-Review Mode

### Option 1: Environment Variable (Temporary)
```bash
SELF_REVIEW_MODE=true bash run_test_agents_directly.sh
```

### Option 2: Update .env File (Permanent)
Edit `agent_ai/Newer stuff/usaco/white_agent/.env`:
```bash
# Enable the mode
SELF_REVIEW_MODE=true

# Optionally adjust number of review rounds
MAX_REVIEW_ROUNDS=3
```

### Option 3: Combine with Other Modes
```bash
# With explanation mode
EXPLAIN_MODE=true SELF_REVIEW_MODE=true bash run_test_agents_directly.sh

# With self-test mode (review first, then test)
SELF_REVIEW_MODE=true SELF_TEST_MODE=true bash run_test_agents_directly.sh

# With all modes
EXPLAIN_MODE=true SELF_REVIEW_MODE=true SELF_TEST_MODE=true GENERATE_TESTS_MODE=true bash run_test_agents_directly.sh
```
