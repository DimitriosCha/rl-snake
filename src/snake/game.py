# game.py
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pygame # type: ignore
import random

from .config import (
    WIDTH, HEIGHT, CELL_SIZE, GRID_W, GRID_H,
    BG, GREEN, RED, TEXT,
    UP, DOWN, LEFT, RIGHT,
    CFG,
)

# ---------- Helpers ----------
def spawn_food(snake: List[Tuple[int, int]]) -> Tuple[int, int]:
    while True:
        fx = random.randrange(GRID_W)
        fy = random.randrange(GRID_H)
        if (fx, fy) not in snake:
            return (fx, fy)

def draw_cell(screen: pygame.Surface, gx: int, gy: int, color: Tuple[int, int, int]) -> None:
    rect = pygame.Rect(gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, color, rect)

def is_opposite(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] == -b[0] and a[1] == -b[1]

# ---------- State ----------
@dataclass
class GameState:
    snake: List[Tuple[int, int]]   # head at index 0
    direction: Tuple[int, int]
    pending: Tuple[int, int]
    food: Tuple[int, int]
    score: int
    last_move: int                 # ms timestamp of last step
    speed_ms: int                  # current step interval
    is_rl: bool         # is this an RL game?

def new_game_state(now_ms: int, is_rl: bool) -> GameState:
    snake = [
        (GRID_W // 2, GRID_H // 2),
        (GRID_W // 2 - 1, GRID_H // 2),
        (GRID_W // 2 - 2, GRID_H // 2),
    ]
    food = spawn_food(snake)
    return GameState(
        snake=snake,
        direction=RIGHT,
        pending=RIGHT,
        food=food,
        score=0,
        last_move=now_ms,
        speed_ms=CFG.move_every_ms,
        is_rl=False,
    )

# ---------- Input / Update / Draw ----------
def handle_input(state: GameState) -> bool:
    """Process events; update pending direction (no 180° turns). Return False to quit."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            cand: Optional[Tuple[int, int]] = None
            if event.key == pygame.K_UP:    cand = UP
            elif event.key == pygame.K_DOWN: cand = DOWN
            elif event.key == pygame.K_LEFT: cand = LEFT
            elif event.key == pygame.K_RIGHT:cand = RIGHT
            if cand is not None and not is_opposite(cand, state.direction):
                state.pending = cand
    return True

def step_game(state: GameState, now_ms: int) -> bool:
    """
    Advance the game by one tick.
    - If is_rl == False: respect real-time pacing (move every speed_ms).
    - If is_rl == True: move every call (no timing gate).
    Returns True if alive, False if game over.
    """
    # Only gate on time in human/pygame mode
    if (not state.is_rl) and (now_ms - state.last_move < state.speed_ms):
        return True  # not time to move yet

    # Commit direction once per tick
    state.direction = state.pending

    hx, hy = state.snake[0]
    dx, dy = state.direction
    nx, ny = hx + dx, hy + dy

    # Wall collision
    if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
        return False

    new_head = (nx, ny)

    # Self collision
    if new_head in state.snake:
        return False

    # Move / grow
    if new_head == state.food:
        state.snake.insert(0, new_head)
        state.score += 1
        # speed up every N foods (kept even in RL; optional)
        if state.score % CFG.foods_per_speedup == 0:
            state.speed_ms = max(CFG.min_move_ms, state.speed_ms - 10)
        state.food = spawn_food(state.snake)
    else:
        state.snake.insert(0, new_head)
        state.snake.pop()

    # In RL, now_ms might be a dummy—still safe to update
    state.last_move = now_ms
    return True

def draw_game(screen: pygame.Surface, font: pygame.font.Font, state: GameState) -> None:
    screen.fill(BG)
    # food
    draw_cell(screen, state.food[0], state.food[1], RED)
    # snake
    for x, y in state.snake:
        draw_cell(screen, x, y, GREEN)
    # score
    txt = font.render(f"Score: {state.score}", True, TEXT)
    screen.blit(txt, (8, 6))

def draw_game_over(screen: pygame.Surface, font: pygame.font.Font, score: int) -> None:
    # Dim with translucent overlay
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 140))  # RGBA
    screen.blit(overlay, (0, 0))

    title = font.render("GAME OVER", True, (240, 240, 250))
    sub   = font.render("Press R to restart", True, (220, 220, 230))
    sco   = font.render(f"Score: {score}", True, (220, 220, 230))

    tx = title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 16))
    sx = sub.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 16))
    cx = sco.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 44))

    screen.blit(title, tx)
    screen.blit(sub, sx)
    screen.blit(sco, cx)