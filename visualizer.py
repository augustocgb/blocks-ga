import pygame
import time
from game_logic import GRID_SIZE, EMPTY_CELL_COLOR
from simulate import get_best_game_history

pygame.init()

CELL_SIZE = 40
GRID_LINE_WIDTH = 1
BOARD_WIDTH = GRID_SIZE * CELL_SIZE
BOARD_HEIGHT = GRID_SIZE * CELL_SIZE
SCORE_AREA_HEIGHT = 40
PIECES_AREA_HEIGHT = 80

SCREEN_WIDTH = BOARD_WIDTH
SCREEN_HEIGHT = BOARD_HEIGHT + SCORE_AREA_HEIGHT + PIECES_AREA_HEIGHT

BACKGROUND_COLOR = (20, 20, 20)
GRID_COLOR = (50, 50, 50)
TEXT_COLOR = (220, 220, 220)

def draw_grid(surface, grid):
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell_rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, grid[r][c], cell_rect)
            pygame.draw.rect(surface, GRID_COLOR, cell_rect, GRID_LINE_WIDTH)

def draw_piece(surface, piece_data, x, y, cell_size):
    """Draw a single piece"""
    for r_offset, c_offset in piece_data["coords"]:
        rect = pygame.Rect(
            x + c_offset * cell_size,
            y + r_offset * cell_size,
            cell_size - 1,
            cell_size - 1
        )
        pygame.draw.rect(surface, piece_data["color"], rect)

def draw_available_pieces(surface, pieces_per_move, move_idx):
    """Draw the available pieces for the current state"""
    if not pieces_per_move or move_idx >= len(pieces_per_move):
        return
    
    available_pieces = pieces_per_move[move_idx]
    piece_area_y = BOARD_HEIGHT + SCORE_AREA_HEIGHT
    pygame.draw.rect(surface, (45, 45, 55), (0, piece_area_y, SCREEN_WIDTH, PIECES_AREA_HEIGHT))
    
    small_cell_size = 20
    spacing = SCREEN_WIDTH // (len(available_pieces) + 1)
    
    for i, piece in enumerate(available_pieces):
        if not piece["placed"]:
            draw_piece(surface, piece["piece_data"], 
                      spacing * (i + 1) - small_cell_size * 2,
                      piece_area_y + PIECES_AREA_HEIGHT//4,
                      small_cell_size)

def draw_score(surface, score):
    font = pygame.font.SysFont('Arial', 24)
    score_text = font.render(f'Score: {score}', True, TEXT_COLOR)
    text_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, BOARD_HEIGHT + SCORE_AREA_HEIGHT // 2))
    surface.blit(score_text, text_rect)

def visualize_best_game():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Best Game Replay")
    clock = pygame.time.Clock()

    history = get_best_game_history()
    if not history:
        print("No game history available!")
        return

    current_state = 0
    running = True
    paused = True
    delay = 500
    last_update = pygame.time.get_ticks()
    scores = history['scores']
    pieces_per_move = history['available_pieces_per_move']

    while running:
        current_time = pygame.time.get_ticks()

        # Handle continuous actions for held keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            if current_time - last_update >= delay:
                current_state = min(len(history['grid_states']) - 1, current_state + 1)
                last_update = current_time
        if keys[pygame.K_LEFT]:
            if current_time - last_update >= delay:
                current_state = max(0, current_state - 1)
                last_update = current_time
        if keys[pygame.K_UP]:
            if current_time - last_update >= 100:  # Faster response for delay adjustment
                delay = max(100, delay - 100)
                last_update = current_time
        if keys[pygame.K_DOWN]:
            if current_time - last_update >= 100:
                delay += 100
                last_update = current_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused and current_time - last_update >= delay:
            if current_state < len(history['grid_states']) - 1:
                current_state += 1
                last_update = current_time

        screen.fill(BACKGROUND_COLOR)
        draw_grid(screen, history['grid_states'][current_state])
        draw_score(screen, scores[current_state])
        draw_available_pieces(screen, pieces_per_move, current_state)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    visualize_best_game()