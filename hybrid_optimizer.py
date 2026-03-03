from genetic_ai import GeneticAlgorithm
from gradient_ai import GradientDescentAI
from simulate import evaluate_chromosome, get_best_game_history, reset_best_tracking
import time

class HybridOptimizer:
    def __init__(self,
                 chromosome_length=6,
                 ga_pop_size=30,
                 ga_generations=15,
                 sgd_iterations=20,
                 games_per_eval_ga=5,
                 games_per_eval_sgd=50,
                 games_per_perturbation_sgd=10,
                 sgd_lr=0.01,
                 sgd_perturbation=0.01,
                 sgd_momentum=0.5):
        
        self.chromosome_length = chromosome_length
        self.ga_pop_size = ga_pop_size
        self.ga_generations = ga_generations
        self.sgd_iterations = sgd_iterations
        self.games_per_eval_ga = games_per_eval_ga
        self.games_per_eval_sgd = games_per_eval_sgd
        self.games_per_perturbation_sgd = games_per_perturbation_sgd
        
        # SGD Hyperparameters for fine-tuning (usually lower LR/perturbation than from-scratch SGD)
        self.sgd_lr = sgd_lr
        self.sgd_perturbation = sgd_perturbation
        self.sgd_momentum = sgd_momentum

    def run(self):
        print("="*60)
        print("STARTING HYBRID OPTIMIZATION".center(60))
        print("="*60)
        print(f"Phase 1: Genetic Algorithm (Global Search)")
        print(f"  - Population: {self.ga_pop_size}")
        print(f"  - Generations: {self.ga_generations}")
        print(f"  - Games/Eval: {self.games_per_eval_ga}")
        
        reset_best_tracking()
        
        def evaluate_ga(chrom, num_games=self.games_per_eval_ga, seed=None):
            return evaluate_chromosome(chrom, num_games=num_games, seed=seed)

        ga = GeneticAlgorithm(
            population_size=self.ga_pop_size,
            chromosome_length=self.chromosome_length,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elitism_count=2
        )

        best_ga_ind, best_ga_fitness_ind, (ga_gens, ga_best, ga_avg, ga_best_fitness, _, ga_chrom_hist) = ga.run_evolution(evaluate_ga, self.ga_generations)
        
        print("\n" + "-"*60)
        print(f"Phase 1 Complete.")
        print(f"Best GA Score: {best_ga_ind.best_score}")
        print(f"Best GA Weights: {[f'{w:.3f}' for w in best_ga_ind.chromosome]}")
        print("-"*60 + "\n")

        print(f"Phase 2: SGD Fine-Tuning (Local Optimization)")
        print(f"  - Starting Weights: Best from GA")
        print(f"  - Iterations: {self.sgd_iterations}")
        print(f"  - Games/Eval: {self.games_per_eval_sgd}")
        print(f"  - Games/Perturbation: {self.games_per_perturbation_sgd}")
        print(f"  - Learning Rate: {self.sgd_lr}")
        
        def evaluate_sgd(chrom, num_games=self.games_per_eval_sgd, seed=None):
            return evaluate_chromosome(chrom, num_games=num_games, seed=seed)

        sgd = GradientDescentAI(
            chromosome_length=self.chromosome_length,
            learning_rate=self.sgd_lr,
            perturbation_size=self.sgd_perturbation,
            momentum=self.sgd_momentum,
            initial_weights=best_ga_ind.chromosome
        )
        
        best_hybrid_chrom, (sgd_iters, sgd_best, sgd_avg, sgd_chrom_hist) = sgd.train(
            evaluate_sgd, 
            self.sgd_iterations,
            num_games_per_eval=self.games_per_eval_sgd,
            num_games_per_perturbation=self.games_per_perturbation_sgd
        )

        print("\n" + "="*60)
        print("HYBRID OPTIMIZATION COMPLETE".center(60))
        print("="*60)
        print(f"Final Best Score: {sgd.best_score_all_time}")
        print(f"Final Weights: {[f'{w:.3f}' for w in best_hybrid_chrom]}")
        
        actual_ga_gens = len(ga_gens)
        sgd_iters_shifted = [i + actual_ga_gens for i in sgd_iters]
        
        return {
            'ga_stats': (ga_gens, ga_best, ga_avg, ga_best_fitness, ga_chrom_hist),
            'sgd_stats': (sgd_iters_shifted, sgd_best, sgd_avg, sgd_chrom_hist),
            'best_score': sgd.best_score_all_time,
            'best_weights': best_hybrid_chrom,
            'best_ga_ind': best_ga_ind,
            'best_ga_fitness_ind': best_ga_fitness_ind
        }
