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

def simulate_game(chromosome):
    """Simulates a full game using the given chromosome for decisions."""
    global best_score, best_game_history
    
    grid = [[EMPTY_CELL_COLOR for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    score = 0
    streak = 0
    
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
        # Generate exactly 3 new pieces for this round
        current_pieces = []
        for _ in range(3):
            piece = random.choice(SHAPES).copy()
            current_pieces.append({
                "piece_data": piece,
                "placed": False
            })

        game_history['moves'].append(None)
        game_history['grid_states'].append(copy.deepcopy(grid))
        game_history['available_pieces_per_move'].append(copy.deepcopy(current_pieces))
        game_history['scores'].append(score)

        pieces_placed = 0
        while pieces_placed < 3 and not game_over:

            # Get unplaced pieces
            unplaced = [p for p in current_pieces if not p["placed"]]
            move = ai_player.choose_move(grid, unplaced)
            if move is None:
                game_over = True
                break

            # Mark the used piece as placed
            piece_data = move["piece_data"]
            for piece in current_pieces:
                if piece["piece_data"] == piece_data:
                    piece["placed"] = True
                    current_pieces.remove(piece)
                    break

            # Apply the move
            for r_offset, c_offset in piece_data["coords"]:
                grid[move["target_row"] + r_offset][move["target_col"] + c_offset] = piece_data["color"]

            # Update score for placing piece
            score += len(piece_data["coords"])

            game_history['moves'].append({
                'piece_data': piece_data,
                'target_row': move["target_row"],
                'target_col': move["target_col"]
            })
            game_history['grid_states'].append(copy.deepcopy(grid))
            game_history['available_pieces_per_move'].append(copy.deepcopy(current_pieces))
            game_history['scores'].append(score)

            # Check for line clears
            rows_cleared = 0
            cols_cleared = 0
            for r in range(GRID_SIZE):
                if all(grid[r][c] != EMPTY_CELL_COLOR for c in range(GRID_SIZE)):
                    rows_cleared += 1
                    for c in range(GRID_SIZE):
                        grid[r][c] = EMPTY_CELL_COLOR
            for c in range(GRID_SIZE):
                if all(grid[r][c] != EMPTY_CELL_COLOR for r in range(GRID_SIZE)):
                    cols_cleared += 1
                    for r in range(GRID_SIZE):
                        grid[r][c] = EMPTY_CELL_COLOR

            total_cleared = rows_cleared + cols_cleared
            if total_cleared > 0:
                streak += total_cleared
                score += (streak * 18)
            else:
                streak = 0

            if total_cleared > 0:
                # Record everything after the clear
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

    #print("score:", score)
    return score

def evaluate_chromosome(chromosome, num_games=3):
    total_score = 0
    best_score = 0

    for game_num in range(num_games):
        score = simulate_game(chromosome)

        if score > best_score:
            best_score = score

        total_score += score

    return int(total_score / num_games), best_score

def get_best_game_history():
    return best_game_history