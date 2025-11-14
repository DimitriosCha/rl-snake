# src/rl/agents/dqn.py

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Tuple

import random
import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore


# =========================
#  Device Utility
# =========================

def get_device() -> torch.device:
    """
    Get the best available device:
    - Metal GPU (MPS) on macOS with M-series chip
    - CUDA on systems with NVIDIA GPU
    - CPU fallback
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# =========================
#  Q-Network
# =========================

class QNetwork(nn.Module):
    """
    Simple MLP that maps state -> Q-values for each action.
    """

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, obs_dim]
        return self.net(x)  # [B, n_actions]


# =========================
#  Replay Buffer
# =========================

class ReplayBuffer:
    """
    Fixed-size buffer that stores (s, a, r, s', done) transitions,
    and allows random minibatch sampling.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[
            Tuple[np.ndarray, int, float, np.ndarray, bool]
        ] = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),  # [B, obs_dim]
            np.array(actions, dtype=np.int64),  # [B]
            np.array(rewards, dtype=np.float32),  # [B]
            np.array(next_states, dtype=np.float32),  # [B, obs_dim]
            np.array(dones, dtype=np.float32),  # [B]
        )

    def __len__(self) -> int:
        return len(self.buffer)


# =========================
#  DQN Config
# =========================

@dataclass
class DQNConfig:
    # Discount factor
    gamma: float = 0.99

    # Optimizer / learning
    lr: float = 1e-3
    batch_size: int = 64

    # Replay buffer
    replay_size: int = 50_000
    start_training_after: int = 500  # #env steps before we start gradient updates
    train_every: int = 1  # gradient step every N env steps

    # Target network update
    target_update_freq: int = 1_000  # copy weights every N env steps

    # Exploration (epsilon-greedy)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 5_000  # linear decay over this many steps

    # Device (auto-detect best available: MPS > CUDA > CPU)
    device: torch.device | None = None

    def __post_init__(self):
        """Set device to best available if not explicitly provided."""
        if self.device is None:
            self.device = get_device()


# =========================
#  DQN Agent
# =========================

class DQNAgent:
    """
    Deep Q-Network agent that:
      - chooses actions with epsilon-greedy over Q(s, a)
      - stores transitions in replay buffer
      - periodically trains the Q-network using Bellman targets
      - uses a target network for stable Q-targets
    """

    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig | None = None):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.cfg = cfg or DQNConfig()

        # Networks
        self.policy_net = QNetwork(obs_dim, n_actions).to(self.cfg.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.cfg.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.cfg.replay_size)

        # Global step counter (counts env steps / action selections)
        self.global_step: int = 0

    # ---------- Exploration schedule ----------

    def compute_epsilon(self) -> float:
        """
        Linearly decay epsilon from epsilon_start to epsilon_end
        over epsilon_decay_steps.
        """
        frac = min(self.global_step / float(self.cfg.epsilon_decay_steps), 1.0)
        eps = self.cfg.epsilon_start - frac * (
            self.cfg.epsilon_start - self.cfg.epsilon_end
        )
        return eps

    # ---------- Public API used by train.py ----------

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.
        - state: np.ndarray of shape [obs_dim]
        Returns:
        - action: int in [0, n_actions)
        """
        epsilon = self.compute_epsilon()
        self.global_step += 1

        # Exploration
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)

        # Exploitation
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.cfg.device
        ).unsqueeze(0)  # [1, obs_dim]

        with torch.no_grad():
            q_values = self.policy_net(state_t)  # [1, n_actions]
            action = int(torch.argmax(q_values, dim=1).item())

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def maybe_train(self) -> float | None:
        """
        Perform a single DQN gradient step if:
          - we have enough transitions in replay buffer AND
          - the global_step meets the train_every condition.

        Returns:
          - loss value (float) if a training step was performed
          - None otherwise
        """
        # Wait until buffer has enough experiences
        if len(self.replay_buffer) < max(
            self.cfg.start_training_after, self.cfg.batch_size
        ):
            return None

        # Train only every N steps to save compute
        if self.global_step % self.cfg.train_every != 0:
            return None

        # Sample a minibatch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.cfg.batch_size
        )

        # Convert to tensors
        device = self.cfg.device
        states_t = torch.tensor(states, dtype=torch.float32, device=device)  # [B, obs_dim]
        actions_t = torch.tensor(actions, dtype=torch.int64, device=device)  # [B]
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)  # [B]
        next_states_t = torch.tensor(
            next_states, dtype=torch.float32, device=device
        )  # [B, obs_dim]
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)  # [B]

        # Current Q(s, a) for the actions taken
        q_values = self.policy_net(states_t)  # [B, n_actions]
        q_sa = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # [B]

        # Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t)  # [B, n_actions]
            max_next_q = next_q_values.max(dim=1)[0]  # [B]
            targets = rewards_t + self.cfg.gamma * (1.0 - dones_t) * max_next_q  # [B]

        # Loss = MSE(Q(s,a), target)
        loss = nn.functional.mse_loss(q_sa, targets)

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target network
        if self.global_step % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    # ---------- Greedy action for play mode ----------

    def act_greedy(self, state: np.ndarray) -> int:
        """
        Deterministic action: argmax_a Q(s, a).
        Ignores epsilon / exploration. Use this for 'watch the agent play'.
        """
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.cfg.device
        ).unsqueeze(0)  # [1, obs_dim]

        with torch.no_grad():
            q_values = self.policy_net(state_t)
            action = int(torch.argmax(q_values, dim=1).item())

        return action

    # ---------- Save / Load ----------

    def save(self, path: str) -> None:
        """
        Save only the policy network parameters.
        """
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load parameters into both policy and target networks.
        """
        state_dict = torch.load(path, map_location=self.cfg.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
