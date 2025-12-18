# USACO Agent Testing & AgentBeats Integration

This project contains USACO (USA Computing Olympiad) competitive programming agents that can be tested locally and integrated with the AgentBeats platform.

## Project Structure

```
usaco_agentbeats/
├── README.md                           # This file
├── Notes - Using the agentbeats... .pdf  # AgentBeats platform guide
├── run_test_agents_directly.sh         # Main test script
├── test_agents_directly.py             # Python test harness
├── LLM_INTEGRATION_GUIDE.md            # Detailed LLM setup guide
├── AGENTBEATS_REGISTRATION_GUIDE.md    # Platform registration guide
├── final_data_subset/                  # Test data
│   ├── usaco_questions.json            # Problem definitions
│   └── tests/                          # Test cases per problem
├── logs/                               # Runtime logs
└── agents/
    ├── green_agent/                    # Assessor agent (poses problems, evaluates solutions)
    │   ├── tools.py                    # Agent tools
    │   ├── .env                        # Configuration
    │   └── .venv/                      # Python virtual environment
    └── white_agent/                    # Assessee agent (solves problems using LLM)
        ├── tools.py                    # Agent tools with LLM integration
        ├── .env                        # Configuration & feature flags
        └── .venv/                      # Python virtual environment
```

---

## Quick Start: Running Tests Locally

### Running the Test Script

```bash
bash run_test_agents_directly.sh
```

### What the Script Does

1. **Green Agent** provides the problem statement and sample test
2. **White Agent** generates a solution using the LLM
3. **Green Agent** evaluates the solution against all test cases
4. Results are logged to `logs/<problem_id>.txt`

---


## LLM Integration

The white agent uses LiteLLM to call LLM APIs. Configure your API key in: `agents/white_agent/.env` by setting

OPENAI_API_KEY=sk-proj-xxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxx


### Command Options

```bash
# Use default problem (735_bronze_the_lost_cow)
bash run_test_agents_directly.sh

# Random bronze-level problem
bash run_test_agents_directly.sh bronze

# Random silver-level problem
bash run_test_agents_directly.sh silver

# Random gold-level problem
bash run_test_agents_directly.sh gold

# Random platinum-level problem
bash run_test_agents_directly.sh platinum
```


### Feature Flags

Configure these in `agents/white_agent/.env`:

| Flag | Default | Description |
|------|---------|-------------|
| `EXPLAIN_MODE` | `false` | Generate solution plan and explanations |
| `SELF_TEST_MODE` | `false` | Test solution against sample, fix if wrong |
| `SELF_REVIEW_MODE` | `false` | LLM reviews and improves its own code |
| `GENERATE_TESTS_MODE` | `false` | Generate test cases and validate solution |
| `MAX_FIX_ATTEMPTS` | `3` | Max attempts to fix failing solutions |
| `MAX_REVIEW_ROUNDS` | `2` | Max self-review iterations |
| `NUM_GENERATED_TESTS` | `3` | Number of test cases to generate |


More details documented in white_agent_features/
---

## AgentBeats Platform Integration

### Overview

AgentBeats is a competitive AI agent evaluation platform. Your agents run locally and connect via Cloudflare tunnels.

- **Green Agent (Assessor)**: Poses problems and evaluates solutions
- **White Agent (Assessee)**: Solves problems using LLM

### Integration Steps

1. **Set up Cloudflare Tunnel** (for public access to local agents)
2. **Start your agents locally**
3. **Register agents on v2.agentbeats.org**
4. **Create and run assessments**

### Starting Agents for AgentBeats

**Start Green Agent:**
```bash
cd "agents/green_agent"
source .venv/bin/activate
agentbeats run green_agent_card.toml --launcher_port 8000 --agent_port 8001
```

**Start White Agent:**
```bash
cd "agents/white_agent"
source .venv/bin/activate
agentbeats run white_agent_card.toml --launcher_port 8002 --agent_port 8003
```

### Agent URLs (after Cloudflare tunnel setup)

- Green Agent: `https://usacogreen.imn.it.com`
- White Agent: `https://usacowhite.imn.it.com`

### Registration on AgentBeats

1. Go to https://v2.agentbeats.org
2. Login with GitHub
3. Click "+" to add an agent
4. Fill in:
   - **Name**: Your agent name
   - **Deploy Type**: Remote
   - **Is Assessor**: Check for green agent, uncheck for white agent
   - **Controller URL**: Your Cloudflare tunnel URL
5. Create Assessment by selecting your green (assessor) and white (assessee) agents

For detailed registration steps with screenshots, see:
- `Notes - Using the agentbeats v2 platform - 2025.11 (1).pdf`
- `AGENTBEATS_REGISTRATION_GUIDE.md`

---

## Agent Architecture

### Green Agent (Assessor)

**Tools:**
- `get_problem()` - Returns the USACO problem statement
- `get_sample_test()` - Returns sample input/output
- `evaluate_solution(code)` - Runs code against all test cases

**Configuration:**
- Default problem: `735_bronze_the_lost_cow`
- Override with `USACO_PROBLEM_ID` environment variable
- Time limit: 4 seconds per test case

### White Agent (Assessee)

**Tools:**
- `request_problem_from_green_agent(problem)` - Receives problem
- `solve_problem()` - Generates solution using LLM
- `get_my_solution()` - Main entry point for evaluation
- `get_buggy_solution()` - Returns intentionally wrong solution (for testing)

**Features:**
- Automatic sample extraction from problem statements
- Multi-step reasoning with chain-of-thought
- Self-testing and automatic fix attempts
- Self-review for code quality
- Generated test case validation

---


