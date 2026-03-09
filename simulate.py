from game_logic import (
    EMPTY_CELL_COLOR, GRID_SIZE, SHAPES,
    count_holes_and_blockades,
    get_aggregate_height_and_bumpiness,
    count_potential_lines_cleared,
    count_contact_points,
    get_all_valid_moves
)
from genetic_ai import Individual
import random
import copy

best_score = 0
best_game_history = None

def reset_best_tracking():
    global best_score, best_game_history
    best_score = 0
    best_game_history = None

def simulate_game(chromosome, render_callback=None):
    """Simulates a full game using the given chromosome."""
    global best_score, best_game_history
    
    grid = [[EMPTY_CELL_COLOR for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    score, streak = 0, 0
    
    game_history = {
        'moves': [],
        'grid_states': [],
        'available_pieces_per_move': [],
        'scores': [],
        'final_score': None
    }
    
    ai_player = Individual(chromosome_length=len(chromosome))
    ai_player.chromosome = chromosome
    
    game_over = False
    
    while not game_over:
        current_pieces = []
        for _ in range(3):
            piece = random.choice(SHAPES).copy()
            current_pieces.append({"piece_data": piece, "placed": False})

        game_history['moves'].append(None)
        game_history['grid_states'].append(copy.deepcopy(grid))
        game_history['available_pieces_per_move'].append(copy.deepcopy(current_pieces))
        game_history['scores'].append(score)

        pieces_placed = 0
        while pieces_placed < 3 and not game_over:
            unplaced = [p for p in current_pieces if not p["placed"]]
            move = ai_player.choose_move(grid, unplaced, streak)
            if move is None:
                game_over = True
                break

            piece_data = move["piece_data"]
            for piece in current_pieces:
                if piece["piece_data"] == piece_data:
                    piece["placed"] = True
                    current_pieces.remove(piece)
                    break

            for r_offset, c_offset in piece_data["coords"]:
                grid[move["target_row"] + r_offset][move["target_col"] + c_offset] = piece_data["color"]

            score += len(piece_data["coords"])

            if render_callback:
                render_callback(grid, score, current_pieces)

            game_history['moves'].append({
                'piece_data': piece_data,
                'target_row': move["target_row"],
                'target_col': move["target_col"]
            })
            game_history['grid_states'].append(copy.deepcopy(grid))
            game_history['available_pieces_per_move'].append(copy.deepcopy(current_pieces))
            game_history['scores'].append(score)

            rows_cleared, cols_cleared = 0, 0
            for r in range(GRID_SIZE):
                if all(grid[r][c] != EMPTY_CELL_COLOR for c in range(GRID_SIZE)):
                    rows_cleared += 1
                    for c in range(GRID_SIZE): grid[r][c] = EMPTY_CELL_COLOR
            for c in range(GRID_SIZE):
                if all(grid[r][c] != EMPTY_CELL_COLOR for r in range(GRID_SIZE)):
                    cols_cleared += 1
                    for r in range(GRID_SIZE): grid[r][c] = EMPTY_CELL_COLOR

            total_cleared = rows_cleared + cols_cleared
            if total_cleared > 0:
                streak += total_cleared
                score += (streak * 18)
            else:
                streak = 0

            if total_cleared > 0:
                if render_callback:
                    render_callback(grid, score, current_pieces)

                game_history['moves'].append({
                    'piece_data': piece_data,
                    'target_row': move["target_row"],
                    'target_col': move["target_col"]
                })
                game_history['grid_states'].append(copy.deepcopy(grid))
                game_history['available_pieces_per_move'].append(copy.deepcopy(current_pieces))
                game_history['scores'].append(score)

            pieces_placed += 1

    game_history['final_score'] = score
    if score > best_score:
        best_score = score
        best_game_history = game_history

    return score

def evaluate_chromosome(chromosome, num_games=3, seed=None, render_callbacks=None):
    total_score = 0
    best_score = 0

    for game_num in range(num_games):
        if seed is not None:
            random.seed(seed + game_num)
            
        cb = None
        if render_callbacks and game_num < len(render_callbacks):
            cb = render_callbacks[game_num]

        score = simulate_game(chromosome, render_callback=cb)

        if score > best_score:
            best_score = score

        total_score += score

    if seed is not None:
        random.seed()

    return int(total_score / num_games), best_score

def get_best_game_history():
    return best_game_history