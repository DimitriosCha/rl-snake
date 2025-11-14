# src/rl/train.py
from __future__ import annotations
import argparse
import csv
import os
from typing import Tuple

import time
import numpy as np  # type: ignore

from src.rl.env import SnakeRLEnv
from src.rl.policies import policy_random, policy_greedy, policy_eps_greedy
from src.rl.agents.dqn import DQNAgent, DQNConfig


# --------------------------
# Episode loop (non-learning policies)
# --------------------------
def run_episode(env: SnakeRLEnv, policy: str, epsilon: float) -> Tuple[int, float, int]:
    """
    Run a single episode using a fixed (non-learning) policy:
    - random
    - greedy
    - eps-greedy

    Returns:
        steps: number of steps taken
        total: total return (sum of rewards)
        score: final score from env.info["score"] if provided
    """
    obs = env.reset()
    total = 0.0
    steps = 0
    score = 0

    while True:
        if policy == "random":
            a = policy_random(obs, env)
        elif policy == "greedy":
            a = policy_greedy(obs, env)
        elif policy in ("eps-greedy", "epsilon-greedy"):
            a = policy_eps_greedy(obs, env, epsilon)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        obs, r, done, info = env.step(a)
        total += r
        steps += 1

        if done or steps >= 10_000:
            score = info.get("score", 0)
            break

    return steps, total, score


# --------------------------
# DQN training loop
# --------------------------
def train_dqn(env: SnakeRLEnv, episodes: int, out_csv: str) -> str:
    """
    Train a DQN agent on the SnakeRLEnv for a given number of episodes.
    Logs per-episode (steps, return, score) to a CSV file.

    Returns:
        model_path: path to the saved DQN weights.
    """

    # Get observation dimension from a reset
    first_obs = env.reset()
    obs_arr = np.asarray(first_obs, dtype=np.float32)
    obs_dim = int(obs_arr.shape[-1])

    # Infer number of actions from env
    if hasattr(env, "action_space_n"):
        n_actions = int(getattr(env, "action_space_n"))
    elif hasattr(env, "n_actions"):
        n_actions = int(getattr(env, "n_actions"))
    elif hasattr(env, "ACTIONS"):
        # e.g. a dict or list of actions
        n_actions = len(getattr(env, "ACTIONS"))
    else:
        raise AttributeError(
            "Could not infer number of actions from env. "
            "Expected env.action_space_n, env.n_actions, or env.ACTIONS."
        )

    dqn_cfg = DQNConfig()  # auto-picks best device (MPS on your Mac, else CPU)
    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, cfg=dqn_cfg)

    rows = [("ep", "steps", "return", "score")]
    print(f"Training DQN for {episodes} episode(s)")
    print("ep,steps,return,score")

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        score = 0

        while True:
            # Choose action with epsilon-greedy over Q(s, a)
            action = agent.select_action(np.asarray(state, dtype=np.float32))

            # Step environment
            next_state, reward, done, info = env.step(action)

            # Store transition and maybe train
            agent.store_transition(
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
            _loss = agent.maybe_train()  # we don't need to print it yet

            state = next_state
            total_reward += reward
            steps += 1

            if done or steps >= 10_000:
                score = info.get("score", 0)
                break

        print(f"{ep},{steps},{total_reward:.3f},{score}")
        rows.append((ep, steps, float(f"{total_reward:.6f}"), score))

    # Save CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\n[DQN] Saved training results → {out_csv}")

    # Save model weights next to CSV
    model_path = os.path.splitext(out_csv)[0] + "_dqn.pt"
    agent.save(model_path)
    print(f"[DQN] Saved model weights → {model_path}")

    return model_path


# --------------------------
# DQN play loop (greedy, with rendering)
# --------------------------
def play_dqn(checkpoint_path: str, render_delay: float = 0.05) -> None:
    print(f"[PLAY] Using checkpoint: {checkpoint_path}")

    env = SnakeRLEnv(debug=False)

    first_obs = env.reset()
    print("[PLAY] First obs shape:", np.asarray(first_obs).shape)

    obs_arr = np.asarray(first_obs, dtype=np.float32)
    obs_dim = int(obs_arr.shape[-1])

    if hasattr(env, "action_space_n"):
        n_actions = int(getattr(env, "action_space_n"))
    elif hasattr(env, "n_actions"):
        n_actions = int(getattr(env, "n_actions"))
    elif hasattr(env, "ACTIONS"):
        n_actions = len(getattr(env, "ACTIONS"))
    else:
        n_actions = 4

    cfg = DQNConfig()
    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, cfg=cfg)
    agent.load(checkpoint_path)

    print("[PLAY] Loaded DQN weights. Starting episode...")

    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    score = 0

    while not done:
        action = agent.act_greedy(np.asarray(state, dtype=np.float32))

        state, reward, done, info = env.step(action)

        total_reward += reward
        steps += 1
        score = info.get("score", score)

        print(f"[PLAY] step={steps}, reward={reward}, done={done}")  # <--- debug

        env.render()
        time.sleep(render_delay)

    print(f"[PLAY] Episode finished. steps={steps}, return={total_reward:.3f}, score={score}")

    if hasattr(env, "close"):
        env.close()

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)

    # For backward compatibility: policies + new "dqn" mode
    parser.add_argument(
        "--policy",
        type=str,
        default="random",
        choices=["random", "greedy", "eps-greedy", "dqn"],
        help=(
            "Which policy/agent to run:\n"
            "  random, greedy, eps-greedy → non-learning\n"
            "  dqn → train a DQN agent"
        ),
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="epsilon for eps-greedy (ignored for DQN)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/runs",
        help="CSV/model will be saved here",
    )

    # New: play mode & checkpoint
    parser.add_argument(
        "--play",
        action="store_true",
        help="If set, load a trained DQN and watch it play one episode greedily.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to DQN checkpoint (.pt) for --play mode. "
             "If not provided, defaults to <outdir>/rl_dqn_dqn.pt",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, f"rl_{args.policy}.csv")

    # If --play is set, just watch the agent play and exit
    if args.play:
        # Default checkpoint path mirrors where train_dqn() saves it
        checkpoint = args.checkpoint
        if checkpoint is None:
            checkpoint = os.path.splitext(out_csv)[0] + "_dqn.pt"  # e.g. data/runs/rl_dqn_dqn.pt

        play_dqn(checkpoint)
        return

    # Normal modes: training or fixed policies
    env = SnakeRLEnv(debug=False, render_enabled=True)

    if args.policy == "dqn":
        # Use the DQN training loop
        train_dqn(env, args.episodes, out_csv)
    else:
        # Use the existing non-learning episode runner
        print(
            f"Running {args.episodes} episode(s) with "
            f"policy={args.policy} ε={args.epsilon}"
        )
        print("ep,steps,return,score")

        rows = [("ep", "steps", "return", "score")]
        for ep in range(1, args.episodes + 1):
            steps, ret, score = run_episode(env, args.policy, args.epsilon)
            print(f"{ep},{steps},{ret:.3f},{score}")
            rows.append((ep, steps, float(f"{ret:.6f}"), score))

        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"\nSaved results → {out_csv}")


if __name__ == "__main__":
    main()
