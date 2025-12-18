# White Agent Explanation Mode

The white agent supports an **Explanation Mode** that generates detailed plans and explanations while still returning only executable code to the green agent.

## How It Works

When `EXPLAIN_MODE` is enabled, the white agent:

1. **Generates a Solution Plan**
   - Problem understanding
   - Input/output format analysis
   - Algorithm approach
   - Key insights and edge cases
   - Step-by-step implementation plan

2. **Provides Implementation Explanation**
   - Concise explanation of the code approach

3. **Returns Only Executable Code**
   - Plans and explanations are **printed to stdout** (visible in test output)
   - **Only code is returned** to the green agent for evaluation
   - This ensures proper evaluation while providing transparency

## Enabling Explanation Mode

### Option 1: Environment Variable (Temporary)
```bash
EXPLAIN_MODE=true bash run_test_agents_directly.sh
```

### Option 2: Update .env File (Permanent)
Edit `agent_ai/Newer stuff/usaco/white_agent/.env`:
```bash
# Change from false to true
EXPLAIN_MODE=true
```

### Option 3: Set in Shell (Current Session)
```bash
export EXPLAIN_MODE=true
cd /Users/nivasini/Documents/usaco_agent
bash run_test_agents_directly.sh
```


