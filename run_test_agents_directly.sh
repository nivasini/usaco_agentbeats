#!/bin/bash
# Wrapper script to run test_agents_directly.py with proper environment
#
# Usage:
#   bash run_test_agents_directly.sh [difficulty_level]
#
# Arguments:
#   difficulty_level (optional): bronze, silver, gold, or platinum
#                                If specified, randomly selects a problem from that level
#
# Examples:
#   bash run_test_agents_directly.sh           # Use default problem
#   bash run_test_agents_directly.sh bronze    # Random bronze problem
#   bash run_test_agents_directly.sh gold      # Random gold problem

# Change to the directory where this script is located (agent_ai)
cd "$(dirname "$0")"

# Set the OpenAI API key
export OPENAI_API_KEY="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"

# Check if difficulty level argument is provided
DIFFICULTY_LEVEL="${1:-}"

if [ -n "$DIFFICULTY_LEVEL" ]; then
    # Validate difficulty level
    case "$DIFFICULTY_LEVEL" in
        bronze|silver|gold|platinum)
            echo "Selecting random ${DIFFICULTY_LEVEL} problem..."

            # Use Python to select a random problem from the specified difficulty level
            SELECTED_PROBLEM=$(python3 << PYEOF
import json
import random

# Load problems
with open('final_data_subset/usaco_questions.json', 'r') as f:
    data = json.load(f)

# Filter problems by difficulty level
difficulty = '${DIFFICULTY_LEVEL}'
problems = [pid for pid in data.keys() if pid.split('_')[1] == difficulty]

if problems:
    selected = random.choice(problems)
    print(selected)
else:
    print('')
PYEOF
)

            if [ -n "$SELECTED_PROBLEM" ]; then
                export USACO_PROBLEM_ID="$SELECTED_PROBLEM"
                echo "Selected problem: $SELECTED_PROBLEM"
            else
                echo "Error: No ${DIFFICULTY_LEVEL} problems found"
                exit 1
            fi
            ;;
        *)
            echo "Error: Invalid difficulty level '${DIFFICULTY_LEVEL}'"
            echo "Valid options: bronze, silver, gold, platinum"
            echo ""
            echo "Usage: bash run_test_agents_directly.sh [difficulty_level]"
            exit 1
            ;;
    esac
else
    echo "Using default problem (set USACO_PROBLEM_ID to override)"
fi

echo ""

# Activate white agent's venv (has litellm installed)
source "agents/white_agent/.venv/bin/activate"

# Mock agentbeats and run the test
python3 << 'PYEOF'
import sys
import types

# Mock agentbeats before any imports
mock_ab = types.ModuleType('agentbeats')
mock_ab.tool = lambda f: f
sys.modules['agentbeats'] = mock_ab

# Now run the test_agents_directly.py
exec(open('test_agents_directly.py').read())
PYEOF
