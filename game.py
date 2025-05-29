import pygame
import random
import sys

GRID_SIZE = 8
CELL_SIZE = 40
GRID_LINE_WIDTH = 1
BOARD_WIDTH = GRID_SIZE * CELL_SIZE
BOARD_HEIGHT = GRID_SIZE * CELL_SIZE

SCORE_AREA_HEIGHT = 40

PREVIEW_CELL_SIZE_RATIO = 0.65
MAX_PREVIEW_PIECE_DIM_CELLS = 4
PIECE_PREVIEW_AREA_TOP_MARGIN = 25
PIECE_PREVIEW_AREA_BOTTOM_MARGIN = 25
PIECE_PREVIEW_AREA_HEIGHT = (PIECE_PREVIEW_AREA_TOP_MARGIN +
                           MAX_PREVIEW_PIECE_DIM_CELLS * int(CELL_SIZE * PREVIEW_CELL_SIZE_RATIO) +
                           PIECE_PREVIEW_AREA_BOTTOM_MARGIN)


SCREEN_WIDTH = BOARD_WIDTH
SCREEN_HEIGHT = BOARD_HEIGHT + SCORE_AREA_HEIGHT + PIECE_PREVIEW_AREA_HEIGHT

BACKGROUND_COLOR = (20, 20, 20)
GRID_COLOR = (50, 50, 50)
EMPTY_CELL_COLOR = (35, 35, 35)
TEXT_COLOR = (220, 220, 220)
GAME_OVER_BG_COLOR = (0,0,0,220)

# --- Block Shapes ---
# Each shape is a list of (row_offset, col_offset) tuples
# (0,0) is the top-left most cell of the shape's logical bounding box
SHAPES = [
     {"id": 0, "coords": [(0,0)], "color": (192, 192, 192), "name": "dot"},
     {"id": 1, "coords": [(0,0), (0,1)], "color": (90, 200, 200), "name": "1x2"},
     {"id": 2, "coords": [(0,0), (1,0)], "color": (90, 200, 200), "name": "2x1"},
     {"id": 3, "coords": [(0,0), (0,1), (0,2)], "color": (60, 170, 220), "name": "1x3"},
     {"id": 4, "coords": [(0,0), (1,0), (2,0)], "color": (60, 170, 220), "name": "3x1"},
     {"id": 5, "coords": [(0,0), (0,1), (1,0), (1,1)], "color": (220, 220, 70), "name": "2x2_O"},
     {"id": 6, "coords": [(0,0), (1,0), (1,1)], "color": (220, 150, 50), "name": "L3_1"},
     {"id": 7, "coords": [(0,1), (1,1), (1,0)], "color": (220, 150, 50), "name": "L3_2"},
     {"id": 8, "coords": [(0,0), (0,1), (1,1)], "color": (220, 150, 50), "name": "L3_3"},
     {"id": 9, "coords": [(0,0), (0,1), (1,0)], "color": (220, 150, 50), "name": "L3_4"},
     {"id": 10, "coords": [(0,0), (1,0), (2,0), (2,1)], "color": (230, 120, 40), "name": "L4v"},
     {"id": 11, "coords": [(0,1), (1,1), (2,1), (2,0)], "color": (70, 70, 200), "name": "J4"},
     {"id": 12, "coords": [(0,0), (1,0), (2,0), (3,0)], "color": (100, 200, 235), "name": "I4v"},
     {"id": 13, "coords": [(0,0), (0,1), (0,2), (0,3)], "color": (100, 200, 235), "name": "I4h"},
     {"id": 14, "coords": [(0,0), (0,1), (0,2), (1,1)], "color": (190, 70, 190), "name": "T4"},
     {"id": 15, "coords": [(0,1), (0,2), (1,0), (1,1)], "color": (70, 200, 70), "name": "S4h"},
     {"id": 16, "coords": [(0,0), (0,1), (1,1), (1,2)], "color": (200, 70, 70), "name": "Z4h"},
     #{"id": 17, "coords": [(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)], "color": (255,100,187), "name": "3x3hole"},
     {"id": 18, "coords": [(0,0),(0,2),(1,1),(2,0),(2,2)], "color": (255,155,200), "name": "X_5block"},
     {"id": 19, "coords": [(0,0), (1,0), (2,0), (3,0), (4,0)], "color": (100, 200, 235), "name": "I5v"},
     {"id": 20, "coords": [(0,0), (0,1), (0,2), (0,3), (0,4)], "color": (100, 200, 235), "name": "I5h"},
     {"id": 21, "coords": [(1,0), (2,0), (0,1), (1,1)], "color": (70, 200, 70), "name": "S4v"},
     {"id": 22, "coords": [(1,0), (2,0), (1,1), (0,1)], "color": (200, 70, 70), "name": "Z4v"},
     {"id": 23, "coords": [(0,1), (1,1), (2,1), (2,0), (2,2)], "color": (70, 200, 70), "name": "TLongV1"},
     {"id": 24, "coords": [(0,1), (1,1), (2,1), (0,0), (0,2)], "color": (70, 200, 70), "name": "TLongV2"},
     {"id": 25, "coords": [(1,0), (0,0), (1,1), (2,0), (1,2)], "color": (70, 200, 70), "name": "TLongH1"},
     {"id": 26, "coords": [(1,0), (1,1), (1,2), (0,2), (2,2)], "color": (200, 70, 70), "name": "TLongH2"},
     {"id": 27, "coords": [(0,0), (1,1)], "color": (200, 70, 70), "name": "2x2diag1"},
     {"id": 28, "coords": [(1,0), (0,1)], "color": (200, 70, 70), "name": "2x2diag2"},
     {"id": 29, "coords": [(0,0), (1,1), (2,2)], "color": (200, 70, 70), "name": "3x3diag1"},
     {"id": 30, "coords": [(2,0), (1,1), (0,2)], "color": (200, 70, 70), "name": "3x3diag2"},
     {"id": 31, "coords": [(0,1), (1,1), (2,1), (2,0)], "color": (230, 120, 40), "name": "L4v1"}
]

def get_all_valid_moves(grid_data, available_pieces):
    possible_moves = []
    for piece_idx, piece_info in enumerate(available_pieces):
        if not piece_info["placed"]:
            piece_data = piece_info["piece_data"]
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    # Check if piece can be placed at this position
                    valid = True
                    for r_offset, c_offset in piece_data["coords"]:
                        check_r = r + r_offset
                        check_c = c + c_offset
                        if not (0 <= check_r < GRID_SIZE and 0 <= check_c < GRID_SIZE):
                            valid = False
                            break
                        if grid_data[check_r][check_c] != EMPTY_CELL_COLOR:
                            valid = False
                            break
                    
                    if valid:
                        possible_moves.append({
                            'piece_index': piece_idx,
                            'piece_data': piece_data,
                            'target_row': r,
                            'target_col': c
                        })
    return possible_moves

def count_holes_and_blockades(grid_data_state):
    holes = 0
    for c in range(GRID_SIZE):
        has_block = False
        for r in range(GRID_SIZE):
            if grid_data_state[r][c] != EMPTY_CELL_COLOR:
                has_block = True
            elif has_block:
                holes += 1
    return holes

def get_aggregate_height_and_bumpiness(grid_data_state):
    col_heights = []
    for c in range(GRID_SIZE):
        for r in range(GRID_SIZE):
            if grid_data_state[r][c] != EMPTY_CELL_COLOR:
                col_heights.append(GRID_SIZE - r)
                break
        else:
            col_heights.append(0)
    
    aggregate_height = sum(col_heights)
    bumpiness = sum(abs(col_heights[i] - col_heights[i+1]) for i in range(len(col_heights)-1))
    max_height = max(col_heights) if col_heights else 0
    
    return aggregate_height, bumpiness, max_height

def count_potential_lines_cleared(grid_after_move):
    lines = 0
    # Check rows
    for r in range(GRID_SIZE):
        if all(cell != EMPTY_CELL_COLOR for cell in grid_after_move[r]):
            lines += 1
    # Check columns
    for c in range(GRID_SIZE):
        if all(grid_after_move[r][c] != EMPTY_CELL_COLOR for r in range(GRID_SIZE)):
            lines += 1
    return lines

def count_contact_points(grid_before_move, piece_coords, r_target, c_target):
    contacts = 0
    for r_offset, c_offset in piece_coords:
        r, c = r_target + r_offset, c_target + c_offset
        
        # Check cell below
        if r + 1 >= GRID_SIZE:
            contacts += 1  # Floor contact
        elif grid_before_move[r + 1][c] != EMPTY_CELL_COLOR:
            contacts += 1
            
        # Check cell above
        if r - 1 >= 0 and grid_before_move[r - 1][c] != EMPTY_CELL_COLOR:
            contacts += 1
            
        # Check cell to right
        if c + 1 >= GRID_SIZE:
            contacts += 1  # Wall contact
        elif grid_before_move[r][c + 1] != EMPTY_CELL_COLOR:
            contacts += 1
            
        # Check cell to left
        if c - 1 < 0:
            contacts += 1  # Wall contact
        elif grid_before_move[r][c - 1] != EMPTY_CELL_COLOR:
            contacts += 1
            
    return contacts

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pygame Blocks")
clock = pygame.time.Clock()
FONT_SIZE_NORMAL = 24
FONT_SIZE_LARGE = 48
FONT_SIZE_SMALL = 20
FONT_NORMAL = pygame.font.SysFont('Arial', FONT_SIZE_NORMAL)
FONT_GAME_OVER = pygame.font.SysFont('Arial', FONT_SIZE_LARGE, bold=True)
FONT_RESTART = pygame.font.SysFont('Arial', FONT_SIZE_SMALL)

# --- Game State Variables ---
grid_data = []
available_pieces_info = []
score = 0
streak = 0
game_over_flag = False

dragging_piece = None # Dict: {data (shape,color), screen_pos_x, screen_pos_y, original_piece_info_index}
drag_offset_from_piece_topleft_x = 0
drag_offset_from_piece_topleft_y = 0



# --- Helper Functions ---
def get_piece_bounding_box_dims(piece_coords):
    if not piece_coords:
        return 0, 0
    min_r = min(c[0] for c in piece_coords) # Should be 0 if coords are normalized
    max_r = max(c[0] for c in piece_coords)
    min_c = min(c[1] for c in piece_coords) # Should be 0
    max_c = max(c[1] for c in piece_coords)
    return max_r - min_r + 1, max_c - min_c + 1 # height_cells, width_cells

def draw_single_piece(surface, piece_render_data, top_left_x, top_left_y, cell_size_to_use):
    shape_coords = piece_render_data["coords"]
    color = piece_render_data["color"]
    for r_offset, c_offset in shape_coords:
        # Adjust drawing rect to be slightly smaller to show grid lines if they are part of background
        rect_inner = pygame.Rect(
            top_left_x + c_offset * cell_size_to_use,
            top_left_y + r_offset * cell_size_to_use,
            cell_size_to_use - GRID_LINE_WIDTH,
            cell_size_to_use - GRID_LINE_WIDTH
        )
        pygame.draw.rect(surface, color, rect_inner)

def generate_and_setup_new_pieces():
    global available_pieces_info
    infos = []
    
    preview_cell_size = int(CELL_SIZE * PREVIEW_CELL_SIZE_RATIO)
    
    # Calculate total width required for 3 pieces to center them
    total_width_of_previews = 0
    preview_piece_widths_px = []
    chosen_shapes_for_preview = []

    for _ in range(3):
        chosen_shape = random.choice(SHAPES).copy()
        chosen_shapes_for_preview.append(chosen_shape)
        _, w_cells = get_piece_bounding_box_dims(chosen_shape["coords"])
        piece_width_px = w_cells * preview_cell_size
        preview_piece_widths_px.append(piece_width_px)
        total_width_of_previews += piece_width_px

    num_pieces = 3
    spacing = (SCREEN_WIDTH - total_width_of_previews) / (num_pieces + 1)
    if spacing < 10: spacing = 10 # Minimum spacing

    current_x = spacing
    preview_area_base_y = BOARD_HEIGHT + SCORE_AREA_HEIGHT + PIECE_PREVIEW_AREA_TOP_MARGIN

    for i in range(num_pieces):
        piece_data = chosen_shapes_for_preview[i]
        h_cells, w_cells = get_piece_bounding_box_dims(piece_data["coords"])
        
        piece_width_px = w_cells * preview_cell_size
        piece_height_px = h_cells * preview_cell_size
        
        # Center piece vertically in its preview slot (optional, simple top align is fine too)
        y_pos = preview_area_base_y + ( (MAX_PREVIEW_PIECE_DIM_CELLS * preview_cell_size) - piece_height_px) / 2


        draw_rect = pygame.Rect(current_x, y_pos, piece_width_px, piece_height_px)
        # Make clickable area slightly larger than visual piece for easier interaction
        click_expansion = 5 
        clickable_rect = pygame.Rect(current_x - click_expansion, y_pos - click_expansion, 
                                      piece_width_px + 2 * click_expansion, piece_height_px + 2 * click_expansion)

        infos.append({
            "piece_data": piece_data,
            "draw_rect": draw_rect, # For reference, actual drawing uses original_pos
            "click_rect": clickable_rect,
            "preview_cell_size": preview_cell_size,
            "original_pos_x": current_x,
            "original_pos_y": y_pos,
            "placed": False
        })
        current_x += piece_width_px + spacing
    
    available_pieces_info = infos

def draw_game_board(surface):
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell_rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, grid_data[r][c], cell_rect) # Cell color
            pygame.draw.rect(surface, GRID_COLOR, cell_rect, GRID_LINE_WIDTH) # Grid line border

def draw_available_pieces_display(surface):
    preview_area_rect_y = BOARD_HEIGHT + SCORE_AREA_HEIGHT
    pygame.draw.rect(surface, (45,45,55), (0, preview_area_rect_y, SCREEN_WIDTH, PIECE_PREVIEW_AREA_HEIGHT))

    for i, item_info in enumerate(available_pieces_info):
        # Only draw if not placed AND not currently being dragged by player
        if not item_info["placed"] and (dragging_piece is None or dragging_piece["original_piece_info_index"] != i):
            draw_single_piece(surface, item_info["piece_data"],
                              item_info["original_pos_x"], item_info["original_pos_y"],
                              item_info["preview_cell_size"])

def is_move_valid(piece_coords, target_grid_row, target_grid_col):
    for r_offset, c_offset in piece_coords:
        check_r, check_c = target_grid_row + r_offset, target_grid_col + c_offset
        if not (0 <= check_r < GRID_SIZE and 0 <= check_c < GRID_SIZE):
            return False # Piece out of grid bounds
        if grid_data[check_r][check_c] != EMPTY_CELL_COLOR:
            return False # Grid cell is already occupied
    return True

def commit_piece_to_grid(piece_data_to_commit, target_grid_row, target_grid_col):
    global grid_data
    for r_offset, c_offset in piece_data_to_commit["coords"]:
        grid_data[target_grid_row + r_offset][target_grid_col + c_offset] = piece_data_to_commit["color"]

def process_line_clears():
    global grid_data, streak
    
    rows_to_clear_indices = []
    for r_idx in range(GRID_SIZE):
        if all(grid_data[r_idx][c_idx] != EMPTY_CELL_COLOR for c_idx in range(GRID_SIZE)):
            rows_to_clear_indices.append(r_idx)

    cols_to_clear_indices = []
    for c_idx in range(GRID_SIZE):
        if all(grid_data[r_idx][c_idx] != EMPTY_CELL_COLOR for r_idx in range(GRID_SIZE)):
            cols_to_clear_indices.append(c_idx)

    # Clear the identified rows
    for r_idx in rows_to_clear_indices:
        for c_idx in range(GRID_SIZE):
            grid_data[r_idx][c_idx] = EMPTY_CELL_COLOR
    
    # Clear the identified columns
    for c_idx in cols_to_clear_indices:
        for r_idx in range(GRID_SIZE): # Iterate through rows for this column
            grid_data[r_idx][c_idx] = EMPTY_CELL_COLOR
            
    total_lines_cleared = len(rows_to_clear_indices) + len(cols_to_clear_indices)

    # Increment streak if cleared any lines
    if total_lines_cleared == 0:
        streak = 0
    else:
        streak += total_lines_cleared

    return total_lines_cleared

def process_score(piece_data_to_commit):
    global score, streak
    score += (len(piece_data_to_commit["coords"]) + (streak * 18))

def can_any_available_piece_be_placed():
    for item_info in available_pieces_info:
        if not item_info["placed"]: # If piece hasn't been placed yet
            piece_coords_to_check = item_info["piece_data"]["coords"]
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if is_move_valid(piece_coords_to_check, r, c):
                        return True # Found at least one valid move for one piece
    return False # No available piece can be placed anywhere

def draw_current_score(surface):
    score_area_y = BOARD_HEIGHT
    pygame.draw.rect(surface, (50,60,70), (0, score_area_y, SCREEN_WIDTH, SCORE_AREA_HEIGHT)) # BG for score
    score_display_text = FONT_NORMAL.render(f"Score: {score}", True, TEXT_COLOR)
    text_rect = score_display_text.get_rect(center=(SCREEN_WIDTH // 2, score_area_y + SCORE_AREA_HEIGHT // 2))
    surface.blit(score_display_text, text_rect)

def draw_game_over_overlay(surface):
    overlay_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA) # SRCALPHA for transparency
    overlay_surface.fill(GAME_OVER_BG_COLOR)
    surface.blit(overlay_surface, (0,0))

    title_text = FONT_GAME_OVER.render("GAME OVER", True, (255, 60, 60))
    title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
    surface.blit(title_text, title_rect)

    final_score_text = FONT_NORMAL.render(f"Final Score: {score}", True, TEXT_COLOR)
    final_score_rect = final_score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 10))
    surface.blit(final_score_text, final_score_rect)
    
    restart_msg_text = FONT_RESTART.render("Press 'R' to Restart", True, TEXT_COLOR)
    restart_rect = restart_msg_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
    surface.blit(restart_msg_text, restart_rect)

def initialize_or_reset_game():
    global grid_data, score, game_over_flag, dragging_piece, available_pieces_info
    grid_data = [[EMPTY_CELL_COLOR for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    score = 0
    game_over_flag = False
    dragging_piece = None
    generate_and_setup_new_pieces()
    
    # Check if the very first set of pieces can be placed
    if not can_any_available_piece_be_placed():
        game_over_flag = True

# --- Main Game Setup ---
initialize_or_reset_game()

# --- Game Loop ---
running = True
while running:
    current_mouse_pos = pygame.mouse.get_pos()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()

        if game_over_flag:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                initialize_or_reset_game()
            continue # Skip other event processing if game is over

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and dragging_piece is None: # Left click to pick up a piece
                for i, item_info in enumerate(available_pieces_info):
                    if not item_info["placed"] and item_info["click_rect"].collidepoint(current_mouse_pos):
                        dragging_piece = {
                            "data": item_info["piece_data"],
                            "screen_pos_x": current_mouse_pos[0] - (item_info["preview_cell_size"] / 2), # Initial drag pos (center on mouse)
                            "screen_pos_y": current_mouse_pos[1] - (item_info["preview_cell_size"] / 2),
                            "original_piece_info_index": i
                        }
                        # Calculate offset of click relative to the preview piece's top-left corner
                        drag_offset_from_piece_topleft_x = current_mouse_pos[0] - item_info["original_pos_x"]
                        drag_offset_from_piece_topleft_y = current_mouse_pos[1] - item_info["original_pos_y"]
                        break
        
        elif event.type == pygame.MOUSEMOTION:
            if dragging_piece:
                # Scale the drag offset from preview size to full CELL_SIZE
                # This ensures the mouse cursor maintains its relative position on the piece image
                preview_cs = available_pieces_info[dragging_piece["original_piece_info_index"]]["preview_cell_size"]
                scaled_offset_x = (drag_offset_from_piece_topleft_x / preview_cs) * CELL_SIZE
                scaled_offset_y = (drag_offset_from_piece_topleft_y / preview_cs) * CELL_SIZE
                
                dragging_piece["screen_pos_x"] = current_mouse_pos[0] - scaled_offset_x
                dragging_piece["screen_pos_y"] = current_mouse_pos[1] - scaled_offset_y

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and dragging_piece:
                # Attempt to place the piece on the grid
                # Target grid cell is determined by where the top-left of the dragged piece is, rounded to nearest cell
                target_col = round(dragging_piece["screen_pos_x"] / CELL_SIZE)
                target_row = round(dragging_piece["screen_pos_y"] / CELL_SIZE)

                if is_move_valid(dragging_piece["data"]["coords"], target_row, target_col):
                    commit_piece_to_grid(dragging_piece["data"], target_row, target_col)
                    original_index = dragging_piece["original_piece_info_index"]
                    available_pieces_info[original_index]["placed"] = True
                    
                    process_line_clears()

                    process_score(dragging_piece["data"])

                    # Check if all 3 pieces from current set are placed
                    if all(p_info["placed"] for p_info in available_pieces_info):
                        generate_and_setup_new_pieces() # Get new set of 3

                        while can_any_available_piece_be_placed() is False:
                            generate_and_setup_new_pieces()

                    # After placing a piece or getting new ones, check for game over
                    if not can_any_available_piece_be_placed():
                        game_over_flag = True
                
                dragging_piece = None

    # --- Drawing Cycle ---
    screen.fill(BACKGROUND_COLOR)
    draw_game_board(screen)
    draw_current_score(screen)
    draw_available_pieces_display(screen)

    # Draw ghost preview for placing dragged piece
    if dragging_piece:
        potential_col = round(dragging_piece["screen_pos_x"] / CELL_SIZE)
        potential_row = round(dragging_piece["screen_pos_y"] / CELL_SIZE)
        
        can_place_at_ghost = is_move_valid(dragging_piece["data"]["coords"], potential_row, potential_col)
        ghost_alpha = 100 # Semi-transparent
        base_ghost_color = (100,255,100) if can_place_at_ghost else (255,100,100) # Green if valid, Red if not
        
        for r_off, c_off in dragging_piece["data"]["coords"]:
            ghost_cell_r, ghost_cell_c = potential_row + r_off, potential_col + c_off
            if 0 <= ghost_cell_r < GRID_SIZE and 0 <= ghost_cell_c < GRID_SIZE: # Only if ghost cell is on grid
                ghost_rect = pygame.Rect(ghost_cell_c * CELL_SIZE, ghost_cell_r * CELL_SIZE, 
                                        CELL_SIZE, CELL_SIZE)
                # Create a temporary surface for alpha blending the ghost
                temp_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                temp_surface.fill((base_ghost_color[0], base_ghost_color[1], base_ghost_color[2], ghost_alpha))
                # Draw border for ghost cells
                pygame.draw.rect(temp_surface, (GRID_COLOR[0]+20, GRID_COLOR[1]+20, GRID_COLOR[2]+20, ghost_alpha+50), (0,0,CELL_SIZE,CELL_SIZE), GRID_LINE_WIDTH)
                screen.blit(temp_surface, ghost_rect.topleft)

    # Draw the actual piece being dragged (on top of everything else)
    if dragging_piece:
        draw_single_piece(screen, dragging_piece["data"], 
                          dragging_piece["screen_pos_x"], dragging_piece["screen_pos_y"], 
                          CELL_SIZE)

    if game_over_flag:
        draw_game_over_overlay(screen)

    pygame.display.flip()
    clock.tick(60)

# --- End of Program ---
pygame.quit()
sys.exit()