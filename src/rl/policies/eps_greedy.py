# src/rl/policies/eps_greedy.py
import numpy as np # type: ignore
from src.rl.policies.random import policy_random
from src.rl.policies.greedy import policy_greedy


def policy_eps_greedy(obs: np.ndarray, env, epsilon: float = 0.1) -> int:
    """
    Epsilon-greedy policy: with probability epsilon, pick random; else pick greedy.
    """
    if np.random.rand() < epsilon:
        return policy_random(obs, env)
    return policy_greedy(obs, env)
