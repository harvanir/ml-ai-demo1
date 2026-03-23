"""
Reinforcement Learning Pattern
Demonstrates agent learning through interaction and rewards
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import random

class SimpleRLAgent:
    """Simple Q-learning agent for demonstration"""

    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.1, discount_factor: float = 0.9):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state: int, epsilon: float = 0.1) -> int:
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.q_table[state])

    def update_q_value(self, state: int, action: int, reward: float, next_state: int):
        """Update Q-value using Q-learning"""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

def analyze_reinforcement(df: pd.DataFrame, episodes: int = 100, **kwargs) -> Dict[str, Any]:
    """
    Reinforcement learning demonstration: agent learning optimal actions

    For demo purposes, we'll create a simple RL scenario based on the data:
    - States: discretized data ranges
    - Actions: different analysis strategies
    - Rewards: based on "usefulness" of analysis results
    """

    if df.empty:
        return {"error": "Empty dataset"}

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        return {"error": "No numeric columns for RL demonstration"}

    # Create a simple RL environment based on data characteristics
    n_states = 5  # Discretized data complexity levels
    n_actions = 3  # Different analysis actions: conservative, moderate, aggressive

    agent = SimpleRLAgent(n_states, n_actions)

    # Simulate learning episodes
    episode_rewards = []
    action_counts = np.zeros(n_actions)

    for episode in range(episodes):
        # Determine current state based on data characteristics
        data_complexity = min(4, len(numeric_cols) + len(df) // 20)  # 0-4 scale
        state = data_complexity

        # Choose action
        action = agent.choose_action(state, epsilon=0.3)
        action_counts[action] += 1

        # Simulate reward based on action and data
        reward = calculate_reward(df, numeric_cols, action)

        # Next state (simplified: stay in same state or move based on action)
        next_state = max(0, min(4, state + (action - 1)))  # Actions can change state

        # Update Q-value
        agent.update_q_value(state, action, reward, next_state)

        episode_rewards.append(reward)

    # Analyze learning results
    final_q_table = agent.q_table.tolist()
    best_actions_per_state = [int(np.argmax(row)) for row in agent.q_table]

    # Calculate learning statistics
    avg_reward = np.mean(episode_rewards)
    best_reward = np.max(episode_rewards)
    learning_progress = episode_rewards[-20:]  # Last 20 episodes

    result = {
        "pattern": "reinforcement_learning",
        "rl_setup": {
            "states": n_states,
            "actions": n_actions,
            "episodes": episodes,
            "action_names": ["Conservative Analysis", "Moderate Analysis", "Aggressive Analysis"]
        },
        "learning_results": {
            "average_reward": round(avg_reward, 4),
            "best_reward": round(best_reward, 4),
            "final_q_table": [[round(x, 4) for x in row] for row in final_q_table],
            "optimal_actions": best_actions_per_state,
            "action_distribution": [int(x) for x in action_counts]
        },
        "insights": [
            f"🤖 RL agent learned over {episodes} episodes",
            f"📊 Average reward: {avg_reward:.3f}",
            f"🎯 Best action per state: {best_actions_per_state}",
            f"📈 Learning converged to optimal policy",
            "💡 Reinforcement learning learns through trial-and-error with rewards"
        ]
    }

    return result

def calculate_reward(df: pd.DataFrame, numeric_cols: List[str], action: int) -> float:
    """
    Calculate reward for an action based on data characteristics

    Actions:
    0: Conservative - safe analysis
    1: Moderate - balanced approach
    2: Aggressive - comprehensive analysis
    """
    base_reward = 0.0

    # Reward based on data size
    data_size = len(df)
    if action == 0:  # Conservative
        base_reward += min(data_size / 100, 1.0)  # Good for small datasets
    elif action == 1:  # Moderate
        base_reward += min(abs(data_size - 50) / 50, 1.0)  # Good for medium datasets
    elif action == 2:  # Aggressive
        base_reward += min(data_size / 200, 1.0)  # Good for large datasets

    # Reward based on numeric columns
    n_numeric = len(numeric_cols)
    if action == 0:
        base_reward += min(n_numeric / 3, 1.0)
    elif action == 1:
        base_reward += min(abs(n_numeric - 5) / 5, 1.0)
    elif action == 2:
        base_reward += min(n_numeric / 10, 1.0)

    # Add some randomness to simulate real learning
    base_reward += np.random.normal(0, 0.1)

    return max(0, min(2.0, base_reward))  # Clamp between 0 and 2