import random
import game_logic as game

try:
    import msvcrt
except ImportError:
    msvcrt = None

class Individual(object):
    def __init__(self, chromosome_length = 6):
        self.chromosome = [random.uniform(-1.0, 1.0) for _ in range(chromosome_length)]
        self.fitness = 0.0
        self.best_score = 0.0

    def evaluate_move_heuristic(self, grid_before_move, piece_data, r_target, c_target, current_streak):
        """Evaluates a potential move using the individual's chromosome and heuristic functions."""
        temp_grid_after_move = [row[:] for row in grid_before_move]
        
        for r_offset, c_offset in piece_data["coords"]:
            check_r, check_c = r_target + r_offset, c_target + c_offset
            if not (0 <= check_r < game.GRID_SIZE and 0 <= check_c < game.GRID_SIZE and temp_grid_after_move[check_r][check_c] == game.EMPTY_CELL_COLOR):
                 return -float('inf')
            temp_grid_after_move[check_r][check_c] = piece_data['color']

        holes = game.count_holes_and_blockades(temp_grid_after_move)
        agg_height, bump, _ = game.get_aggregate_height_and_bumpiness(temp_grid_after_move)
        lines_cleared_val = game.count_potential_lines_cleared(temp_grid_after_move)
        contacts_val = game.count_contact_points(grid_before_move, piece_data["coords"], r_target, c_target)
        
        score = 0
        if len(self.chromosome) >= 5:
            score += self.chromosome[0] * holes
            score += self.chromosome[1] * agg_height
            score += self.chromosome[2] * bump
            score += self.chromosome[3] * lines_cleared_val
            score += self.chromosome[4] * contacts_val
        
        if len(self.chromosome) >= 6:
            predicted_streak = 0
            if lines_cleared_val > 0:
                predicted_streak = current_streak + lines_cleared_val
            else:
                predicted_streak = 0 # Streak breaks if no lines cleared
            
            score += self.chromosome[5] * predicted_streak
             
        if len(self.chromosome) < 5:
            score = -holes - agg_height - bump + lines_cleared_val * 10 + contacts_val
        return score

    def choose_move(self, current_grid_data, current_available_pieces_info_sim, current_streak=0):
        all_possible_moves = game.get_all_valid_moves(current_grid_data, current_available_pieces_info_sim)
        if not all_possible_moves:
            return None

        best_move_details = None
        best_heuristic_score = -float('inf')

        for move in all_possible_moves:
            heuristic_score = self.evaluate_move_heuristic(
                current_grid_data, 
                move['piece_data'], 
                move['target_row'], 
                move['target_col'],
                current_streak
            )
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
        self.elitism_count = min(elitism_count, population_size - 1)
        self.population = []
        self.best_individual_current = Individual(chromosome_length)  # Best individual in the current generation
        self.best_individual_all_time = Individual(chromosome_length)  # Best individual across all generations (by score)
        self.best_fitness_all_time = Individual(chromosome_length) # Best individual across all generations (by fitness)

    def _create_individual(self):
        return Individual(self.chromosome_length)

    def initialize_population(self):
        self.population = [self._create_individual() for _ in range(self.population_size)]

    def eval_pop_fitness(self, game_function):
        # Reset current best for the generation
        self.best_individual_current = self.population[0]
        
        for i, individual in enumerate(self.population):
            individual.fitness, individual.best_score = game_function(individual.chromosome)

            if individual.best_score > self.best_individual_current.best_score:
                self.best_individual_current = individual

        # Update best individual across all generations (by max score)
        if (self.best_individual_current.best_score > self.best_individual_all_time.best_score):
            self.best_individual_all_time.best_score = self.best_individual_current.best_score
            self.best_individual_all_time.fitness = self.best_individual_current.fitness
            self.best_individual_all_time.chromosome = self.best_individual_current.chromosome[:]
            
        # Update best individual across all generations (by average fitness)
        best_fit_current = max(self.population, key=lambda x: x.fitness)
        if best_fit_current.fitness > self.best_fitness_all_time.fitness:
            self.best_fitness_all_time.best_score = best_fit_current.best_score
            self.best_fitness_all_time.fitness = best_fit_current.fitness
            self.best_fitness_all_time.chromosome = best_fit_current.chromosome[:]

    def select_parents(self):
        return random.choices(
            population=self.population,
            weights=[max(0.1, ind.fitness) for ind in self.population],
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
        return individual

    def new_generation(self):
        # Keep elite individuals
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        new_population = [ind for ind in self.population[:self.elitism_count]]
        
        # Create rest of new population through selection/crossover/mutation
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            child1, child2 = self._crossover_one_point(parent1, parent2)
            
            # Mutate children
            child1 = self._mutate_gaussian(child1)
            child2 = self._mutate_gaussian(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population
        return new_population

    def run_evolution(self, game_function, n_generations):
        print("\nStarting evolution... (Press 's' to stop current phase)")
        self.initialize_population()
        
        best_scores, avg_scores, best_finess, avg_fitness, generations = [], [], [], [], []
        chromosome_history = []
        
        for gen in range(n_generations):
            print(f"\n{'='*60}")
            print(f"Generation {gen + 1}/{n_generations}".center(60))
            
            self.eval_pop_fitness(game_function)
            
            generations.append(gen)
            best_scores.append(self.best_individual_current.best_score)
            avg_scores.append(sum(ind.fitness for ind in self.population) / len(self.population))
            best_finess.append(self.best_fitness_all_time.fitness)
            avg_fitness.append(sum(ind.fitness for ind in self.population) / len(self.population))
            
            chromosome_history.append([ind.chromosome[:] for ind in self.population])
            
            print(f"{'='*60}")
            print(f"Generation Best Fitness:          {max(ind.fitness for ind in self.population):>10}")
            print(f"Generation Best Score:            {self.best_individual_current.best_score:>10}")
            print(f"Average Fitness:                  {avg_scores[-1]:>10.2f}")
            print(f"All-Time Best Fitness:            {self.best_fitness_all_time.fitness:>10}")
            print(f"All-Time Best Score:              {self.best_individual_all_time.best_score:>10}")
            print(f"Best Score Chromosome:            [{', '.join([f'{x:>7.3f}' for x in self.best_individual_all_time.chromosome])}]")
            print(f"Best Fitness Chromosome:          [{', '.join([f'{x:>7.3f}' for x in self.best_fitness_all_time.chromosome])}]")
            print(f"{'='*60}")

            if msvcrt and msvcrt.kbhit():
                if msvcrt.getch().decode('utf-8').lower() == 's':
                    print("\n[STOP] 's' pressed. Finishing GA phase...")
                    break

            self.population = self.new_generation()

        print("\nEvolution completed!")
        
        return self.best_individual_all_time, self.best_fitness_all_time, (generations, best_scores, avg_scores, best_finess, avg_fitness, chromosome_history)


