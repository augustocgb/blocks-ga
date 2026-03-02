import argparse
import sys
from genetic_ai import GeneticAlgorithm
from gradient_ai import GradientDescentAI
from hybrid_optimizer import HybridOptimizer
from simulate import evaluate_chromosome, get_best_game_history, reset_best_tracking
from visualizer import visualize_best_game
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
GA_POPULATION_SIZE = 100
GA_N_GENERATIONS = 100
GAMES_PER_EVAL_GA = 10

CHROMOSOME_LENGTH = 6
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
ELITISM_COUNT = 2

SGD_ITERATIONS = 25
GAMES_PER_PERTURBATION_SGD = 10
GAMES_PER_EVAL_SGD = 10

SGD_LEARNING_RATE = 0.1
SGD_PERTURBATION = 0.1
SGD_MOMENTUM = 0.5

def run_comparison():
    parser = argparse.ArgumentParser(description="Run Block Game Optimization Comparison")
    parser.add_argument('--run-ga', action='store_true', help="Run Genetic Algorithm")
    parser.add_argument('--run-sgd', action='store_true', help="Run Stochastic Gradient Descent")
    parser.add_argument('--run-hybrid', action='store_true', help="Run Hybrid Optimization (GA -> SGD)")
    
    args = parser.parse_args()

    if not args.run_ga and not args.run_sgd and not args.run_hybrid:
        run_ga, run_sgd, run_hybrid = True, True, False
    else:
        run_ga, run_sgd, run_hybrid = args.run_ga, args.run_sgd, args.run_hybrid

    print(f"--- Starting Optimization Run ---")

    ga_results, sgd_results, hybrid_results = None, None, None
    ga_best_ind, ga_best_game_history = None, None
    sgd_best_chrom, sgd_best_game_history, sgd_best_score_val = None, None, 0
    hybrid_best_score_val, hybrid_best_game_history = 0, None

    if run_ga:
        print("\n[GA] Running Genetic Algorithm...")
        reset_best_tracking()
        
        def evaluate_ga(chrom, num_games=GAMES_PER_EVAL_GA, seed=None):
            return evaluate_chromosome(chrom, num_games=num_games, seed=seed)

        ga = GeneticAlgorithm(
            population_size=GA_POPULATION_SIZE,
            chromosome_length=CHROMOSOME_LENGTH,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE,
            elitism_count=ELITISM_COUNT
        )

        ga_best_ind, ga_stats = ga.run_evolution(evaluate_ga, GA_N_GENERATIONS)
        ga_results = ga_stats
        ga_best_game_history = get_best_game_history()
        print(f"GA Best Score Reached: {ga_best_ind.best_score}")

    if run_sgd:
        print("\n[SGD] Running Stochastic Gradient Descent...")
        reset_best_tracking()
        
        def evaluate_sgd(chrom, num_games=GAMES_PER_EVAL_SGD, seed=None):
            return evaluate_chromosome(chrom, num_games=num_games, seed=seed)

        sgd = GradientDescentAI(
            chromosome_length=CHROMOSOME_LENGTH,
            learning_rate=SGD_LEARNING_RATE,
            perturbation_size=SGD_PERTURBATION,
            momentum=SGD_MOMENTUM
        )
        
        sgd_best_chrom, sgd_stats = sgd.train(evaluate_sgd, SGD_ITERATIONS)
        sgd_results = sgd_stats
        sgd_best_score_val = sgd.best_score_all_time
        sgd_best_game_history = get_best_game_history()
        print(f"SGD Best Score Reached: {sgd.best_score_all_time}")

    if run_hybrid:
        print("\n[Hybrid] Running Hybrid Optimizer...")
        optimizer = HybridOptimizer(
            chromosome_length=CHROMOSOME_LENGTH,
            ga_pop_size=GA_POPULATION_SIZE,
            ga_generations=GA_N_GENERATIONS,
            sgd_iterations=SGD_ITERATIONS,
            games_per_eval_ga=GAMES_PER_EVAL_GA,
            games_per_eval_sgd=GAMES_PER_EVAL_SGD,
            sgd_lr=SGD_LEARNING_RATE,
            sgd_perturbation=SGD_PERTURBATION,
            sgd_momentum=SGD_MOMENTUM
        )
        
        hybrid_data = optimizer.run()
        hybrid_results = hybrid_data
        hybrid_best_score_val = hybrid_data['best_score']
        hybrid_best_game_history = get_best_game_history()

    if run_ga or run_sgd or run_hybrid:
        plt.figure(figsize=(12, 6))
        
        if run_ga and ga_results:
            ga_gens, ga_best_scores, ga_avg_scores, _, _ = ga_results
            plt.plot(ga_gens, ga_avg_scores, label='GA Average', color='blue', linestyle='-', alpha=0.5)
            plt.plot(ga_gens, ga_best_scores, label='GA Best', color='blue', linestyle='--', linewidth=2)
        
        if run_sgd and sgd_results:
            sgd_iters, sgd_best_scores, sgd_avg_scores = sgd_results
            plt.plot(sgd_iters, sgd_avg_scores, label='SGD Average', color='red', linestyle='-', alpha=0.5)
            plt.plot(sgd_iters, sgd_best_scores, label='SGD Best', color='red', linestyle='--', linewidth=2)
            
        if run_hybrid and hybrid_results:
            h_ga_gens, h_ga_best, h_ga_avg = hybrid_results['ga_stats']
            h_sgd_iters, h_sgd_best, h_sgd_avg = hybrid_results['sgd_stats']
            
            plt.plot(h_ga_gens, h_ga_avg, label='Hybrid (GA Phase) Avg', color='green', linestyle='-', alpha=0.5)
            plt.plot(h_ga_gens, h_ga_best, label='Hybrid (GA Phase) Best', color='green', linestyle='--', linewidth=2)
            plt.plot(h_sgd_iters, h_sgd_avg, label='Hybrid (SGD Phase) Avg', color='lime', linestyle='-', alpha=0.5)
            plt.plot(h_sgd_iters, h_sgd_best, label='Hybrid (SGD Phase) Best', color='lime', linestyle='--', linewidth=2)

        plt.xlabel('Iteration / Generation')
        plt.ylabel('Score')
        plt.title('Performance Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    print("\n--- Final Results ---")
    if run_ga and ga_best_ind:
        print("Genetic Algorithm Best Score:", ga_best_ind.best_score)
        print("GA Weights:", [f"{w:.3f}" for w in ga_best_ind.chromosome])
        print("-" * 30)
    
    if run_sgd and sgd_best_chrom:
        print("SGD Best Score (All Time):", sgd_best_score_val)
        print("SGD Weights:", [f"{w:.3f}" for w in sgd_best_chrom])
        print("-" * 30)

    if run_hybrid and hybrid_results:
        print("Hybrid Best Score:", hybrid_best_score_val)
        print("Hybrid Weights:", [f"{w:.3f}" for w in hybrid_results['best_weights']])
        print("---")

    # --- Visualization ---
    if run_ga:
        print("\nVisualizing GA Best Game...")
        if ga_best_game_history:
            visualize_best_game(ga_best_game_history, title=f"GA Best Game (Score: {ga_best_ind.best_score})")
        else:
            print("No GA game history found.")

    if run_sgd:
        print("\nVisualizing SGD Best Game...")
        if sgd_best_game_history:
            visualize_best_game(sgd_best_game_history, title=f"SGD Best Game (Score: {sgd_best_score_val})")
        else:
            print("No SGD game history found.")
            
    if run_hybrid:
        print("\nVisualizing Hybrid Best Game...")
        if hybrid_best_game_history:
            visualize_best_game(hybrid_best_game_history, title=f"Hybrid Best Game (Score: {hybrid_best_score_val})")
        else:
            print("No Hybrid game history found.")

if __name__ == "__main__":
    run_comparison()
