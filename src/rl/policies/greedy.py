# src/rl/policies/greedy.py
import numpy as np # type: ignore
from src.snake.config import GRID_W, GRID_H, UP, DOWN, LEFT, RIGHT


def rotate_left(direction):
    """Rotate direction 90° counter-clockwise."""
    dx, dy = direction
    return (-dy, dx)


def rotate_right(direction):
    """Rotate direction 90° clockwise."""
    dx, dy = direction
    return (dy, -dx)


def best_move_toward_food(hx: int, hy: int, fx: int, fy: int):
    """
    Returns a preference ordering of moves that reduce Manhattan distance to food.
    Does NOT check collisions; caller should filter unsafe moves.
    """
    prefs = []
    if fx < hx:
        prefs.append(LEFT)
    elif fx > hx:
        prefs.append(RIGHT)
    if fy < hy:
        prefs.append(UP)
    elif fy > hy:
        prefs.append(DOWN)
    # If same row/col, one axis may not add a move; ensure we consider both axes by adding the orthogonal options last
    # This lets the caller still have options when the primary axis is blocked.
    for d in (UP, DOWN, LEFT, RIGHT):
        if d not in prefs:
            prefs.append(d)
    return prefs  # length 4


def dir_to_action(direction) -> int:
    """Map (dx, dy) to the env action id."""
    from src.rl.env import ACTIONS
    for a, d in ACTIONS.items():
        if d == direction:
            return a
    # Fallback (shouldn't happen)
    return 0


def decode_obs(obs: np.ndarray):
    """
    Matches env._obs() layout (9 dims):
    [hx_n, hy_n, fx_n, fy_n, dx, dy, danger_ahead, danger_left, danger_right]
    We recover integer-ish grid coords by multiplying by (W-1)/(H-1).
    """
    hx_n, hy_n, fx_n, fy_n, dx, dy, dan_f, dan_l, dan_r = obs.tolist()
    return hx_n, hy_n, fx_n, fy_n, int(dx), int(dy), bool(dan_f), bool(dan_l), bool(dan_r)


def policy_greedy(obs: np.ndarray, env, epsilon: float = 0.0) -> int:
    """
    Greedy on food distance with simple safety:
    - prefer actions that reduce Manhattan distance
    - avoid any move flagged dangerous if possible
    - if all preferred moves dangerous, choose any safe move
    - if all moves look dangerous, fall back to random
    """
    hx_n, hy_n, fx_n, fy_n, dx, dy, dan_f, dan_l, dan_r = decode_obs(obs)

    # Convert normalized to grid ints (approx)
    hx = int(round(hx_n * (GRID_W - 1)))
    hy = int(round(hy_n * (GRID_H - 1)))
    fx = int(round(fx_n * (GRID_W - 1)))
    fy = int(round(fy_n * (GRID_H - 1)))

    # Current forward/left/right in absolute (dx,dy) terms
    forward = (dx, dy)
    left = rotate_left(forward)
    right = rotate_right(forward)

    # Which actions correspond to those?
    a_forward = dir_to_action(forward)
    a_left = dir_to_action(left)
    a_right = dir_to_action(right)

    # Build a collision map from obs danger flags (relative to heading)
    danger_map = {
        a_forward: dan_f,
        a_left: dan_l,
        a_right: dan_r,
    }
    # The fourth action (the "back" action / 180° turn) may be illegal per env rules;
    # we'll still compute it and let the env reject 180° reversals by ignoring it.
    all_actions = list(range(env.action_space_n))
    for a in all_actions:
        if a not in danger_map:
            # We don't have a relative danger flag for the "back" action; mark as dangerous to de-prioritize.
            danger_map[a] = True

    # Preferred directions by reducing distance
    prefs = best_move_toward_food(hx, hy, fx, fy)
    pref_actions = [dir_to_action(d) for d in prefs]

    # 1) try safe preferred actions in order
    for a in pref_actions:
        if not danger_map.get(a, True):
            return a

    # 2) otherwise, try any safe non-preferred action
    for a in all_actions:
        if not danger_map.get(a, True):
            return a

    # 3) otherwise, fallback to random (we're boxed in)
    return np.random.randint(env.action_space_n)
