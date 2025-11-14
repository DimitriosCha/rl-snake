# src/rl/env.py
from __future__ import annotations
from dataclasses import dataclass
import random

import numpy as np  # type: ignore
import pygame       # type: ignore

from src.snake.config import CFG, GRID_W, GRID_H, UP, DOWN, LEFT, RIGHT
from src.snake.game import GameState, new_game_state, is_opposite, spawn_food

# -----------------------------------------------------------------------------
# Actions: integers -> grid directions (dx, dy)
# -----------------------------------------------------------------------------
ACTIONS = {
    0: UP,
    1: DOWN,
    2: LEFT,
    3: RIGHT,
}

# -----------------------------------------------------------------------------
# Small geometry helpers
# -----------------------------------------------------------------------------
def _left_of(direction):
    """Rotate a direction 90° CCW."""
    dx, dy = direction
    return (-dy, dx)

def _right_of(direction):
    """Rotate a direction 90° CW."""
    dx, dy = direction
    return (dy, -dx)

def _in_bounds(x: int, y: int) -> bool:
    """Check if a cell is inside the grid."""
    return 0 <= x < GRID_W and 0 <= y < GRID_H

def _would_hit(state: GameState, direction) -> bool:
    """
    Returns True if moving the head 1 cell in 'direction' would result
    in a collision with a wall or the snake's body.
    """
    hx, hy = state.snake[0]
    nx, ny = hx + direction[0], hy + direction[1]
    if not _in_bounds(nx, ny):
        return True
    return (nx, ny) in state.snake

def _manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    """Manhattan (L1) distance on the grid."""
    return abs(ax - bx) + abs(ay - by)

# -----------------------------------------------------------------------------
# Observation function
# -----------------------------------------------------------------------------
def _obs(state: GameState) -> np.ndarray:
    """
    Return a compact 9-D observation vector with enough signal
    to learn purposeful behavior (find & eat food) without visuals.

    Features:
      0: hx_n  - head x normalized in [0, 1]
      1: hy_n  - head y normalized in [0, 1]
      2: fx_n  - food x normalized in [0, 1]
      3: fy_n  - food y normalized in [0, 1]
      4: dx    - current direction x component in {-1, 0, 1}
      5: dy    - current direction y component in {-1, 0, 1}
      6: danger_ahead  - 1.0 if the next cell forward would be fatal
      7: danger_left   - 1.0 if the next cell to the left would be fatal
      8: danger_right  - 1.0 if the next cell to the right would be fatal
    """
    hx, hy = state.snake[0]
    fx, fy = state.food

    # Normalize positions to [0, 1]
    denom_w = max(GRID_W - 1, 1)
    denom_h = max(GRID_H - 1, 1)
    hx_n = hx / denom_w
    hy_n = hy / denom_h
    fx_n = fx / denom_w
    fy_n = fy / denom_h

    # Direction as vector components
    dx, dy = state.direction

    # Immediate hazards relative to orientation
    danger_ahead = float(_would_hit(state, state.direction))
    danger_left  = float(_would_hit(state, _left_of(state.direction)))
    danger_right = float(_would_hit(state, _right_of(state.direction)))

    return np.array(
        [
            hx_n, hy_n, fx_n, fy_n,
            float(dx), float(dy),
            danger_ahead, danger_left, danger_right,
        ],
        dtype=np.float32,
    )

# -----------------------------------------------------------------------------
# RL Environment
# -----------------------------------------------------------------------------
@dataclass
class SnakeRLEnv:
    """
    Minimal Gym-like RL environment for your Snake game.

    Rewards:
      + eat_reward  when food is eaten
      + shaping_coef * (d_before - d_after) per step (closer -> positive)
      + step_penalty per step (tiny negative to discourage dithering)
      + death_reward on death
    """
    step_penalty: float = -0.001
    eat_reward: float   = 1.0
    death_reward: float = -1.0
    shaping_coef: float = 0.01
    seed_value: int     = CFG.seed
    debug: bool         = False  # set True to print shaping distances

    # NEW: enable/disable pygame rendering
    render_enabled: bool = False

    def __post_init__(self):
        # Deterministic RNG for reproducibility
        self.rng = random.Random(self.seed_value)
        np.random.seed(self.seed_value)
        self.state: GameState | None = None

        # --- Rendering state (pygame) ---
        self.screen = None
        self.clock = None
        self.cell_size = 20  # pixels per grid cell

        if self.render_enabled:
            pygame.init()
            width_px = GRID_W * self.cell_size
            height_px = GRID_H * self.cell_size
            self.screen = pygame.display.set_mode((width_px, height_px))
            pygame.display.set_caption("Snake DQN")
            self.clock = pygame.time.Clock()

    # Gym-like API -------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        """
        Start a new episode. Returns the initial observation.
        Note: We create the state with is_rl=True so the game advances
        on every call (no real-time gating).
        """
        if seed is not None:
            self.rng.seed(seed)
            np.random.seed(seed)

        now_ms = 0
        self.state = new_game_state(now_ms, is_rl=True)
        return _obs(self.state)

    def step(self, action: int):
        """
        Apply an action (0..3), advance exactly one grid step, and return:
          (obs, reward, terminated, info)
        """
        assert self.state is not None, "Call reset() first."
        assert action in ACTIONS, f"Invalid action {action}"

        # 1) Translate action to a candidate direction; prevent 180° reversals
        cand = ACTIONS[action]
        if not is_opposite(cand, self.state.direction):
            self.state.pending = cand

        # 2) Distance before move for shaping
        hx, hy = self.state.snake[0]
        fx, fy = self.state.food
        d_before = _manhattan(hx, hy, fx, fy)
        if self.debug:
            print(f"Before: {d_before}")

        # 3) Force exactly one move by jumping time forward enough
        now = self.state.last_move + max(self.state.speed_ms, 1)

        # 4) Compute next head position using the (committed) pending direction
        dx, dy = self.state.pending
        nx, ny = hx + dx, hy + dy

        # 5) Terminal checks (death)
        if not _in_bounds(nx, ny) or (nx, ny) in self.state.snake:
            reward = self.death_reward
            terminated = True
            info = {"reason": "death", "score": self.state.score}
            obs = _obs(self.state)  # final observation before death
            return obs, reward, terminated, info

        # 6) Default step outcome
        new_head = (nx, ny)
        reward = self.step_penalty
        terminated = False

        # 7) Eat / move
        if new_head == self.state.food:
            # Eat & grow
            self.state.snake.insert(0, new_head)
            self.state.score += 1
            self.state.food = spawn_food(self.state.snake)

            # Optional speed-up (kept from your cfg)
            if self.state.score % CFG.foods_per_speedup == 0:
                self.state.speed_ms = max(CFG.min_move_ms, self.state.speed_ms - 10)

            reward += self.eat_reward
        else:
            # Move forward without growth
            self.state.snake.insert(0, new_head)
            self.state.snake.pop()

        # 8) Distance after move for shaping (+ if closer)
        hx2, hy2 = self.state.snake[0]
        d_after = _manhattan(hx2, hy2, self.state.food[0], self.state.food[1])
        if self.debug:
            print(f"After:  {d_after}")

        reward += self.shaping_coef * (d_before - d_after)

        # 9) Finalize the step
        self.state.direction = self.state.pending
        self.state.last_move = now

        obs = _obs(self.state)
        info = {"score": self.state.score}
        return obs, reward, terminated, info

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------
    def render(self, mode: str = "human") -> None:
        """
        Render the current game state using pygame.
        Only does anything if render_enabled=True.
        """
        if not self.render_enabled or self.state is None or self.screen is None:
            return

        # Handle window close events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # Clear screen (black)
        self.screen.fill((0, 0, 0))

        # Draw snake
        snake = self.state.snake  # list[(x,y)]
        if snake:
            head = snake[0]
            body = snake[1:]

            # Body
            for (x, y) in body:
                pygame.draw.rect(
                    self.screen,
                    (0, 180, 0),
                    pygame.Rect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                )

            # Head (brighter green)
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),
                pygame.Rect(
                    head[0] * self.cell_size,
                    head[1] * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                ),
            )

        # Draw food (red)
        fx, fy = self.state.food
        pygame.draw.rect(
            self.screen,
            (220, 0, 0),
            pygame.Rect(
                fx * self.cell_size,
                fy * self.cell_size,
                self.cell_size,
                self.cell_size,
            ),
        )

        pygame.display.flip()

        # Limit FPS so it's actually watchable
        if self.clock is not None:
            self.clock.tick(15)

    def close(self) -> None:
        if self.render_enabled:
            pygame.quit()

    @property
    def action_space_n(self) -> int:
        return 4

    @property
    def observation_space_shape(self):
        # 9 features defined in _obs()
        return (9,)
