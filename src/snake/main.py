# main.py
import pygame # type: ignore
from .config import WIDTH, HEIGHT
from .game import new_game_state, handle_input, step_game, draw_game, draw_game_over

def main():
    pygame.init()
    font = pygame.font.SysFont(None, 24)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake — modular")
    clock = pygame.time.Clock()

    state = new_game_state(pygame.time.get_ticks(), is_rl=False)
    running = True

    while running:
        # 1) input
        running = handle_input(state)
        if not running:
            break

        # 2) update
        now = pygame.time.get_ticks()
        alive = step_game(state, now)
        if not alive:
            # draw final frame with overlay
            draw_game(screen, font, state)
            draw_game_over(screen, font, state.score)
            pygame.display.flip()

            # wait for R or Quit
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        state = new_game_state(pygame.time.get_ticks())
                        waiting = False
                clock.tick(30)
            continue

        # 3) render
        draw_game(screen, font, state)
        pygame.display.flip()
        clock.tick(60)  # high FPS; movement gated inside step_game

    pygame.quit()

if __name__ == "__main__":
    main()