from dataclasses import dataclass
import random

# ----- Window & grid -----
WIDTH, HEIGHT = 600, 600
CELL_SIZE = 20
GRID_W, GRID_H = WIDTH // CELL_SIZE, HEIGHT // CELL_SIZE

# ----- Colors -----
BG   = (20, 20, 24)
GREEN = (80, 200, 80)
RED   = (200, 70, 70)
TEXT  = (220, 220, 230)

# ----- Directions (dx, dy) -----
UP, DOWN, LEFT, RIGHT = (0, -1), (0, 1), (-1, 0), (1, 0)

# ----- Tunables (what you'd tweak for difficulty) -----
@dataclass
class Config:
    seed: int = 0
    move_every_ms: int = 120
    foods_per_speedup: int = 5
    min_move_ms: int = 60

CFG = Config(seed=0)

# Make randomness reproducible for debugging
random.seed(CFG.seed)