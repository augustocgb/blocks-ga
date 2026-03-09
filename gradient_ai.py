import random
import copy

try:
    import msvcrt
except ImportError:
    msvcrt = None

class GradientDescentAI:
    def __init__(self, chromosome_length=5, learning_rate=0.1, perturbation_size=0.1, momentum=0.9, initial_weights=None):
        self.chromosome_length = chromosome_length
        self.learning_rate = learning_rate
        self.perturbation_size = perturbation_size
        self.momentum = momentum
        
        if initial_weights is not None:
            if len(initial_weights) != chromosome_length:
                raise ValueError(f"Length of initial_weights ({len(initial_weights)}) must match chromosome_length ({chromosome_length})")
            self.chromosome = list(initial_weights) # Copy to avoid reference issues
        else:
            # Initialize weights with smaller variance to avoid strong initial bias
            self.chromosome = [random.uniform(-0.2, 0.2) for _ in range(chromosome_length)]
            
        self.velocity = [0.0] * chromosome_length
        
        self.best_score_all_time = 0
        self.best_chromosome_all_time = self.chromosome[:]
        
        self.best_avg_score_all_time = 0.0
        self.best_avg_chromosome_all_time = self.chromosome[:]

    def calculate_gradient(self, game_function, num_games_per_perturbation):
        gradient = [0.0] * self.chromosome_length
        eval_seed = random.randint(0, 1000000)
        
        for i in range(self.chromosome_length):
            w_plus = self.chromosome[:]
            w_minus = self.chromosome[:]
            
            w_plus[i] += self.perturbation_size
            w_minus[i] -= self.perturbation_size
            
            score_plus, score_plus_best = game_function(w_plus, num_games=num_games_per_perturbation, seed=eval_seed, is_eval=False)
            score_minus, score_minus_best = game_function(w_minus, num_games=num_games_per_perturbation, seed=eval_seed, is_eval=False)
            
            if score_plus_best > self.best_score_all_time:
                self.best_score_all_time = score_plus_best
                self.best_chromosome_all_time = w_plus[:]
                
            if score_minus_best > self.best_score_all_time:
                self.best_score_all_time = score_minus_best
                self.best_chromosome_all_time = w_minus[:]
                
            if score_plus > self.best_avg_score_all_time:
                self.best_avg_score_all_time = score_plus
                self.best_avg_chromosome_all_time = w_plus[:]
                
            if score_minus > self.best_avg_score_all_time:
                self.best_avg_score_all_time = score_minus
                self.best_avg_chromosome_all_time = w_minus[:]
            
            gradient[i] = (score_plus - score_minus) / (2 * self.perturbation_size)
            
        return gradient

    def train(self, game_function, n_iterations, num_games_per_eval=10, num_games_per_perturbation=10, on_iter_end=None):
        print(f"\nStarting SGD Optimization (Momentum: {self.momentum})... (Press 's' to stop current phase)")
        
        history_iterations, history_avg_scores, history_best_scores, chromosome_history = [], [], [], []
        
        current_avg, current_best = game_function(self.chromosome, num_games=num_games_per_eval, is_eval=True)
        self.best_score_all_time = current_best
        self.best_chromosome_all_time = self.chromosome[:]
        self.best_avg_score_all_time = current_avg
        self.best_avg_chromosome_all_time = self.chromosome[:]
        
        # Record initial inherited performance (Iteration 0)
        history_iterations.append(0)
        history_avg_scores.append(current_avg)
        history_best_scores.append(current_best)
        chromosome_history.append(self.chromosome[:])
        
        if on_iter_end:
            on_iter_end(0, self.chromosome, current_avg, current_best)
        
        for iteration in range(n_iterations):
            print(f"\n{'='*60}")
            print(f"SGD Iteration {iteration + 1}/{n_iterations}".center(60))
            
            print("Calculating gradient...")
            grad = self.calculate_gradient(game_function, num_games_per_perturbation=num_games_per_perturbation) 
            
            grad_magnitude = sum(g**2 for g in grad) ** 0.5
            if grad_magnitude > 0:
                normalized_grad = [g / grad_magnitude for g in grad]
            else:
                normalized_grad = [0] * self.chromosome_length

            for i in range(self.chromosome_length):
                self.velocity[i] = (self.momentum * self.velocity[i]) + (self.learning_rate * normalized_grad[i])
                self.chromosome[i] += self.velocity[i]
                
            print(f"Gradient (Norm):  [{', '.join([f'{g:>7.3f}' for g in normalized_grad])}]")
            print(f"Velocity:         [{', '.join([f'{v:>7.3f}' for v in self.velocity])}]")
            
            avg_score, best_score = game_function(self.chromosome, num_games=num_games_per_eval, is_eval=True)
            
            if best_score > self.best_score_all_time:
                self.best_score_all_time = best_score
                self.best_chromosome_all_time = self.chromosome[:]
                
            if avg_score > self.best_avg_score_all_time:
                self.best_avg_score_all_time = avg_score
                self.best_avg_chromosome_all_time = self.chromosome[:]
                
            history_iterations.append(iteration + 1)
            history_avg_scores.append(avg_score)
            history_best_scores.append(best_score)
            chromosome_history.append(self.chromosome[:])
            
            if on_iter_end:
                on_iter_end(iteration + 1, self.chromosome, avg_score, best_score)
            
            print(f"{'='*60}")
            print(f"Avg Score (New):                  {avg_score:>10.2f}")
            print(f"Best Score (New):                 {best_score:>10}")
            print(f"All-Time Best Avg Score:          {self.best_avg_score_all_time:>10.2f}")
            print(f"All-Time Best Score:              {self.best_score_all_time:>10}")
            print(f"Weights:                          [{', '.join([f'{x:>7.3f}' for x in self.chromosome])}]")
            print(f"{'='*60}")

            if msvcrt and msvcrt.kbhit():
                if msvcrt.getch().decode('utf-8').lower() == 's':
                    print("\n[STOP] 's' pressed. Finishing SGD phase...")
                    break

        return self.best_chromosome_all_time, self.best_avg_chromosome_all_time, (history_iterations, history_best_scores, history_avg_scores, chromosome_history)
