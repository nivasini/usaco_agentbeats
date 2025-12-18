#!/usr/bin/env python3
"""
USACO Green Agent - Entry Point
Runs the AgentBeats controller for the USACO evaluator agent.
"""

import os
import agentbeats as ab

# Import tools
from tools import get_problem, get_sample_test, evaluate_solution

def main():
    """Start the agent controller."""
    # Load agent configuration from TOML
    agent_card_path = os.path.join(os.path.dirname(__file__), "green_agent_card.toml")

    print("Starting USACO Green Agent...")
    print(f"Agent card: {agent_card_path}")

    # Run the controller
    ab.run_ctrl(agent_card_path)

if __name__ == "__main__":
    main()
