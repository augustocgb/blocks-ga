import random
import copy

class GradientDescentAI:
    def __init__(self, chromosome_length=5, learning_rate=0.1, perturbation_size=0.1):
        self.chromosome_length = chromosome_length
        self.learning_rate = learning_rate
        self.perturbation_size = perturbation_size
        # Initialize weights randomly between -1 and 1, similar to GA
        self.chromosome = [random.uniform(-1.0, 1.0) for _ in range(chromosome_length)]
        self.best_score_all_time = 0
        self.best_chromosome_all_time = self.chromosome[:]

    def calculate_gradient(self, game_function, num_games_per_eval=5):
        """
        Estimates the gradient of the score function with respect to the chromosome weights
        using Central Difference Approximation.
        """
        gradient = [0.0] * self.chromosome_length
        
        # Baseline score (optional, for logging/comparison, but not strictly needed for central difference)
        # current_avg_score, _ = game_function(self.chromosome, num_games=num_games_per_eval)

        for i in range(self.chromosome_length):
            # Create perturbed chromosomes
            w_plus = self.chromosome[:]
            w_minus = self.chromosome[:]
            
            w_plus[i] += self.perturbation_size
            w_minus[i] -= self.perturbation_size
            
            # Evaluate both
            score_plus, _ = game_function(w_plus, num_games=num_games_per_eval)
            score_minus, _ = game_function(w_minus, num_games=num_games_per_eval)
            
            # Central difference: df/dx ~ (f(x+h) - f(x-h)) / 2h
            gradient[i] = (score_plus - score_minus) / (2 * self.perturbation_size)
            
        return gradient

    def train(self, game_function, n_iterations):
        print("\nStarting SGD Optimization...")
        
        history_iterations = []
        history_avg_scores = []
        history_best_scores = []
        
        # Initial Evaluation
        current_avg, current_best = game_function(self.chromosome, num_games=5)
        self.best_score_all_time = current_best
        
        for iteration in range(n_iterations):
            print(f"\n{'='*60}")
            print(f"SGD Iteration {iteration + 1}/{n_iterations}".center(60))
            
            # Calculate Gradient
            print("Calculating gradient (this may take a moment)...")
            grad = self.calculate_gradient(game_function, num_games_per_eval=5)
            
            # Normalize Gradient (Gradient Clipping/Scaling)
            # This is crucial because scores can vary wildly, and we want stable updates.
            grad_magnitude = sum(g**2 for g in grad) ** 0.5
            if grad_magnitude > 0:
                normalized_grad = [g / grad_magnitude for g in grad]
            else:
                normalized_grad = [0] * self.chromosome_length

            # Update Weights
            # w = w + learning_rate * normalized_gradient
            # We use + because we want to MAXIMIZE the score (Gradient Ascent)
            # We use normalized gradient to treat learning_rate as a fixed step size.
            
            old_chromosome = self.chromosome[:]
            for i in range(self.chromosome_length):
                self.chromosome[i] += self.learning_rate * normalized_grad[i]
                
            print(f"Gradient:      [{', '.join([f'{g:>7.2f}' for g in grad])}]")
            print(f"Step Vector:   [{', '.join([f'{self.learning_rate * ng:>7.3f}' for ng in normalized_grad])}]")
            
            # Evaluate New Weights
            avg_score, best_score = game_function(self.chromosome, num_games=10)
            
            # Update All-Time Best
            if best_score > self.best_score_all_time:
                self.best_score_all_time = best_score
                self.best_chromosome_all_time = self.chromosome[:]
                
            history_iterations.append(iteration)
            history_avg_scores.append(avg_score)
            history_best_scores.append(best_score)
            
            print(f"{ '='*60}")
            print(f"Avg Score (New):                  {avg_score:>10.2f}")
            print(f"Best Score (New):                 {best_score:>10}")
            print(f"All-Time Best Score:              {self.best_score_all_time:>10}")
            print(f"Weights:                          [{', '.join([f'{x:>7.3f}' for x in self.chromosome])}]")
            print(f"{ '='*60}")

        return self.best_chromosome_all_time, (history_iterations, history_best_scores, history_avg_scores)
