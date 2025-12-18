# White Agent Self-Test Mode

The white agent supports a **Self-Test Mode** that automatically tests solutions against sample inputs and iteratively fixes them if they fail.

## How It Works

When `SELF_TEST_MODE` is enabled, the white agent:

1. **Generates Initial Solution**
   - Uses standard or explanation mode (depending on EXPLAIN_MODE flag)

2. **Tests Against Sample**
   - Runs the solution with the sample input
   - Compares actual output vs expected output

3. **Fixes If Needed**
   - If test fails: LLM analyzes the error and generates a fix
   - If test passes: Returns the working solution
   - Maximum attempts: `MAX_FIX_ATTEMPTS` (default: 3)

4. **Returns Final Solution**
   - Always returns only executable code to green agent
   - Test results and fix attempts printed to stdout

## Enabling Self-Test Mode

### Option 1: Environment Variable (Temporary)
```bash
SELF_TEST_MODE=true bash run_test_agents_directly.sh
```

### Option 2: Update .env File (Permanent)
Edit `agent_ai/Newer stuff/usaco/white_agent/.env`:
```bash
# Change from false to true
SELF_TEST_MODE=true

# Optionally adjust max attempts
MAX_FIX_ATTEMPTS=5
```

### Option 3: Combine with Explanation Mode
```bash
EXPLAIN_MODE=true SELF_TEST_MODE=true bash run_test_agents_directly.sh
```

