import random
import game_logic as game

class Individual(object):
    def __init__(self, chromosome_length = 5):
        self.chromosome = [random.uniform(-1.0, 1.0) for _ in range(chromosome_length)]
        self.fitness = 0.0

    def evaluate_move_heuristic(self, grid_before_move, piece_data, r_target, c_target):
        """Evaluates a potential move using the individual's chromosome and heuristic functions."""
        # 1. Create a temporary grid state if the piece were placed
        temp_grid_after_move = [row[:] for row in grid_before_move]
        
        # Simulate placing the piece (assuming is_move_valid was already checked)
        for r_offset, c_offset in piece_data["coords"]:
            check_r, check_c = r_target + r_offset, c_target + c_offset

            # This check is redundant if called only for valid moves, but good for safety
            if not (0 <= check_r < game.GRID_SIZE and 0 <= check_c < game.GRID_SIZE and temp_grid_after_move[check_r][check_c] == game.EMPTY_CELL_COLOR):
                 return -float('inf')
            temp_grid_after_move[check_r][check_c] = piece_data['color']

        # 2. Calculate heuristic values for the new board state
        holes = game.count_holes_and_blockades(temp_grid_after_move)
        agg_height, bump, _ = game.get_aggregate_height_and_bumpiness(temp_grid_after_move) # Max height also available
        lines_cleared_val = game.count_potential_lines_cleared(temp_grid_after_move)
        contacts_val = game.count_contact_points(grid_before_move, piece_data["coords"], r_target, c_target)
        
        # 3. Calculate score using chromosome weights
        # Remember: more holes = bad, more height/bumpiness = bad. So weights for these might often be negative.
        # More lines cleared = good, more contacts = generally good.
        score = 0
        if len(self.chromosome) == 5:
            score += self.chromosome[0] * holes  # Typically, want to minimize holes, so w0 might be negative
            score += self.chromosome[1] * agg_height # Minimize aggregate height
            score += self.chromosome[2] * bump     # Minimize bumpiness
            score += self.chromosome[3] * lines_cleared_val # Maximize lines cleared
            score += self.chromosome[4] * contacts_val    # Maximize contacts
        else:
            score = -holes - agg_height - bump + lines_cleared_val * 10 + contacts_val
        return score

    def choose_move(self, current_grid_data, current_available_pieces_info_sim):
        all_possible_moves = game.get_all_valid_moves(current_grid_data, current_available_pieces_info_sim)

        if not all_possible_moves:
            return None

        best_move_details = None
        best_heuristic_score = -float('inf')

        for move in all_possible_moves:
            # move = {'piece_index': int, 'piece_data': dict, 'target_row': int, 'target_col': int}
            heuristic_score = self.evaluate_move_heuristic(current_grid_data, move['piece_data'], move['target_row'], move['target_col'])

            if heuristic_score > best_heuristic_score:
                best_heuristic_score = heuristic_score
                best_move_details = move
        
        return best_move_details
        
class GeneticAlgorithm:
    def __init__(self, population_size, chromosome_length=5, mutation_rate=0.1, crossover_rate=0.7, elitism_count=2):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        
        if self.elitism_count >= self.population_size:
            self.elitism_count = max(0, self.population_size -1)

        self.population = []
    
    def _create_individual(self):
        return Individual(self.chromosome_length)

    def initialize_population(self):
        self.population = [self._create_individual() for _ in range(self.population_size)]

    def eval_pop_fitness(self, game_function):
        for i, individual in enumerate(self.population):

            score = game_function(individual.chromosome)
            individual.fitness = score

            print(f"Individual {i} fitness: {score}")

    def select_parents(self):
        return random.choices(
            population=self.population,
            weights=[ind.fitness for ind in self.population],
            k=2
        )
    
    def _crossover_one_point(self, parent1, parent2):
        child1 = self._create_individual()
        child2 = self._create_individual()
        
        if random.random() < self.crossover_rate and len(parent1.chromosome) > 1:
            point = random.randint(1, len(parent1.chromosome) - 1)
            child1.chromosome = parent1.chromosome[:point] + parent2.chromosome[point:]
            child2.chromosome = parent2.chromosome[:point] + parent1.chromosome[point:]
        else:
            child1.chromosome = parent1.chromosome[:]
            child2.chromosome = parent2.chromosome[:]

        return child1, child2

    def _mutate_gaussian(self, individual, mu=0, sigma=0.2):
        for i in range(len(individual.chromosome)):
            if random.random() < self.mutation_rate:
                individual.chromosome[i] += random.gauss(mu, sigma)
                #individual.chromosome[i] = max(min(individual.chromosome[i], 1.0), -1.0)

        return individual

    def run_generation(self, game_function):
        self.eval_pop_fitness(game_function)
        
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        new_population = self.population[:self.elitism_count]
        
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            child1, child2 = self._crossover_one_point(parent1, parent2)
            
            child1 = self._mutate_gaussian(child1)
            child2 = self._mutate_gaussian(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population
        # Best individual
        return self.population[0]
    
    def run_evolution(self, game_function, n_generations):
        self.initialize_population()
        best_fitness = float('-inf')
        best_individual = None
        
        best_scores = []
        avg_scores = []
        generations = []
        
        for gen in range(n_generations):
            best_of_gen = self.run_generation(game_function)
            print(f"\nGeneration {gen + 1}")
            print(f"Best Fitness: {best_of_gen.fitness}")
            print(f"Best Chromosome: {best_of_gen.chromosome}")
            
            # Track statistics
            generations.append(gen)
            best_scores.append(best_of_gen.fitness)
            avg_scores.append(sum(ind.fitness for ind in self.population) / len(self.population))
            
            if best_of_gen.fitness > best_fitness:
                best_fitness = best_of_gen.fitness
                best_individual = best_of_gen
                print(f"New best fitness found: {best_fitness}")
        
        return best_individual, (generations, best_scores, avg_scores)


