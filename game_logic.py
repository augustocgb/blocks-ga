# --- Constants ---
GRID_SIZE = 8
EMPTY_CELL_COLOR = (35, 35, 35)

# --- Block Shapes ---
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
     {"id": 17, "coords": [(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)], "color": (255,100,187), "name": "3x3hole"},
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