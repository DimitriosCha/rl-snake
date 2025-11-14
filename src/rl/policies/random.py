# src/rl/policies/random.py
import numpy as np # type: ignore


def policy_random(obs: np.ndarray, env, epsilon: float = 0.0) -> int:
    """
    Random policy: pick a uniformly random action.
    Expected results most likely inferior to other policies, and often negative.
    """
    return np.random.randint(env.action_space_n)
