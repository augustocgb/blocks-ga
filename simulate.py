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

def simulate_game(chromosome):
    grid = [[EMPTY_CELL_COLOR for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    score = 0
    streak = 0
    
    # Create AI player with the chromosome
    ai_player = Individual(chromosome_length=len(chromosome))
    ai_player.chromosome = chromosome
    
    game_over = False
    while not game_over:
        
        # Generate 3 new pieces
        available_pieces = []
        for _ in range(3):
            piece = random.choice(SHAPES).copy()
            available_pieces.append({
                "piece_data": piece,
                "placed": False
            })
        
        # Try to place all 3 pieces
        for piece_info in available_pieces:
            if game_over:
                break
                
            move = ai_player.choose_move(grid, available_pieces)
            
            if move is None:
                game_over = True
                break
            
            piece_data = move["piece_data"]
            r_target = move["target_row"]
            c_target = move["target_col"]
            
            for r_offset, c_offset in piece_data["coords"]:
                grid[r_target + r_offset][c_target + c_offset] = piece_data["color"]
            
            piece_info["placed"] = True
            
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
            else:
                streak = 0
            
            score += (len(piece_data["coords"]) + (streak * 18))

    return score

def evaluate_chromosome(chromosome, num_games=3):
    total_score = 0

    for _ in range(num_games):
        score = simulate_game(chromosome)
        total_score += score

    return int(total_score / num_games)