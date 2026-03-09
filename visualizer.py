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

def draw_grid(surface, grid, offset_x=0, offset_y=0, cell_size=CELL_SIZE):
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell_rect = pygame.Rect(offset_x + c * cell_size, offset_y + r * cell_size, cell_size, cell_size)
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

def draw_available_pieces(surface, pieces_per_move, move_idx, offset_x=0, offset_y=0, width=SCREEN_WIDTH, height=PIECES_AREA_HEIGHT):
    """Draw the available pieces for the current state"""
    if not pieces_per_move or move_idx >= len(pieces_per_move):
        return
    
    available_pieces = pieces_per_move[move_idx]
    pygame.draw.rect(surface, (45, 45, 55), (offset_x, offset_y, width, height))
    
    small_cell_size = width // 15
    spacing = width // (len(available_pieces) + 1)
    
    for i, piece in enumerate(available_pieces):
        if not piece["placed"]:
            draw_piece(surface, piece["piece_data"], 
                      offset_x + spacing * (i + 1) - small_cell_size * 2,
                      offset_y + height//4,
                      small_cell_size)

def draw_score(surface, score, font, center_x, center_y):
    score_text = font.render(f'Score: {score}', True, TEXT_COLOR)
    text_rect = score_text.get_rect(center=(center_x, center_y))
    surface.blit(score_text, text_rect)

class RealtimeGridVisualizer:
    def __init__(self, rows=1, cols=1, delay_ms=10):
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        self.rows = rows
        self.cols = cols
        self.delay_ms = delay_ms
        self.is_stopped = False
        
        # Scale down cells if there are multiple grids to fit on screen
        self.scale = min(1.0, 1000 / (cols * SCREEN_WIDTH), 800 / (rows * SCREEN_HEIGHT))
        self.cell_size = max(10, int(CELL_SIZE * self.scale))
        
        self.board_w = GRID_SIZE * self.cell_size
        self.board_h = GRID_SIZE * self.cell_size
        self.score_h = max(20, int(SCORE_AREA_HEIGHT * self.scale))
        self.pieces_h = max(40, int(PIECES_AREA_HEIGHT * self.scale))
        
        self.grid_w = self.board_w
        self.grid_h = self.board_h + self.score_h + self.pieces_h

        self.screen_width = self.cols * self.grid_w
        self.screen_height = self.rows * self.grid_h
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(f"Realtime Evaluation Visualizer | Delay: {self.delay_ms}ms | Up/Down: Speed | S: Stop")
        self.font = pygame.font.SysFont('Arial', max(12, int(24 * self.scale)))
        
        self.screen.fill(BACKGROUND_COLOR)
        pygame.display.flip()
        self.last_update = pygame.time.get_ticks()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.is_stopped = True
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.delay_ms = max(0, self.delay_ms - 5)
                    if pygame.get_init():
                        pygame.display.set_caption(f"Realtime Evaluation Visualizer | Delay: {self.delay_ms}ms | Up/Down: Speed | S: Stop")
                elif event.key == pygame.K_DOWN:
                    self.delay_ms += 5
                    if pygame.get_init():
                        pygame.display.set_caption(f"Realtime Evaluation Visualizer | Delay: {self.delay_ms}ms | Up/Down: Speed | S: Stop")
                elif event.key == pygame.K_s:
                    pygame.quit()
                    self.is_stopped = True
                    return False
        return True

    def update_cell(self, row, col, grid, score, available_pieces):
        if not pygame.get_init() or self.is_stopped: return
        
        offset_x = col * self.grid_w
        offset_y = row * self.grid_h
        
        # Clear area
        pygame.draw.rect(self.screen, BACKGROUND_COLOR, (offset_x, offset_y, self.grid_w, self.grid_h))
        
        # Draw board
        draw_grid(self.screen, grid, offset_x, offset_y, self.cell_size)
        
        # Draw score
        draw_score(self.screen, score, self.font, offset_x + self.grid_w // 2, offset_y + self.board_h + self.score_h // 2)
        
        # Draw pieces
        if available_pieces:
            draw_available_pieces(self.screen, [available_pieces], 0, offset_x, offset_y + self.board_h + self.score_h, self.grid_w, self.pieces_h)
            
        pygame.display.update(pygame.Rect(offset_x, offset_y, self.grid_w, self.grid_h))
        
        current_time = pygame.time.get_ticks()
        if current_time - self.last_update < self.delay_ms:
            pygame.time.delay(self.delay_ms - (current_time - self.last_update))
        self.last_update = pygame.time.get_ticks()
        self.handle_events()

def visualize_best_game(history=None, title="Best Game Replay"):
    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()
        
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    
    game_font = pygame.font.SysFont('Arial', 24)

    if history is None:
        history = get_best_game_history()
        
    if not history:
        print("No game history available!")
        return

    current_state = 0
    running = True
    paused = True
    delay = 50
    last_update = pygame.time.get_ticks()
    scores = history['scores']
    pieces_per_move = history['available_pieces_per_move']

    while running:
        current_time = pygame.time.get_ticks()

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
            if current_time - last_update >= 100:  
                delay = max(10, delay - 20)
                last_update = current_time
        if keys[pygame.K_DOWN]:
            if current_time - last_update >= 100:
                delay += 20
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
        draw_score(screen, scores[current_state], game_font, SCREEN_WIDTH // 2, BOARD_HEIGHT + SCORE_AREA_HEIGHT // 2)
        draw_available_pieces(screen, pieces_per_move, current_state)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    visualize_best_game()