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

final_chromosome_test_count = 50

SGD_ITERATIONS = 25
GAMES_PER_PERTURBATION_SGD = 10
GAMES_PER_EVAL_SGD = 33

SGD_LEARNING_RATE = 0.01
SGD_PERTURBATION = 0.01
SGD_MOMENTUM = 0.5

def plot_chromosome_distribution(chromosome_history, generations, title):
    if not chromosome_history or not chromosome_history[0]: return

    normalized_history = []
    for gen_pop in chromosome_history:
        if gen_pop and isinstance(gen_pop[0], (float, int)):
            normalized_history.append([gen_pop])
        else:
            normalized_history.append(gen_pop)
    chromosome_history = normalized_history
        
    n_weights = len(chromosome_history[0][0])
    fig, axes = plt.subplots(n_weights, 1, figsize=(10, 2 * n_weights), sharex=True)
    if n_weights == 1: axes = [axes]
    
    for w in range(n_weights):
        data = []
        for gen_pop in chromosome_history:
            weight_vals = [chrom[w] for chrom in gen_pop]
            data.append(weight_vals)
        
        axes[w].boxplot(data, positions=generations, widths=0.6, manage_ticks=False)
        axes[w].set_ylabel(f'Weight {w}')
        axes[w].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Generation / Iteration')
    fig.suptitle(f'Chromosome Distribution Over Time: {title}')
    plt.tight_layout()

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

        ga_best_ind, ga_best_fitness_ind, ga_stats = ga.run_evolution(evaluate_ga, GA_N_GENERATIONS)
        ga_results = ga_stats
        
        print("\n[Final Evaluation] Testing best GA weights (Score-based) 50 times...")
        final_ga_avg, final_ga_best = evaluate_chromosome(ga_best_ind.chromosome, num_games=50)
        
        print("\n[Final Evaluation] Testing best GA weights (Fitness-based) 50 times...")
        final_ga_fit_avg, final_ga_fit_best = evaluate_chromosome(ga_best_fitness_ind.chromosome, num_games=50)
        
        ga_best_game_history = get_best_game_history()
        ga_best_score_val = ga_best_game_history['final_score'] if ga_best_game_history else 0
        print(f"GA Training Best Score: {ga_best_score_val}")
        print(f"GA Final 50-Game Eval (Score-best) - Avg: {final_ga_avg}, Max: {final_ga_best}")
        print(f"GA Final 50-Game Eval (Fitness-best) - Avg: {final_ga_fit_avg}, Max: {final_ga_fit_best}")

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
        
        sgd_best_chrom, sgd_stats = sgd.train(
            evaluate_sgd, 
            SGD_ITERATIONS,
            num_games_per_eval=GAMES_PER_EVAL_SGD,
            num_games_per_perturbation=GAMES_PER_PERTURBATION_SGD
        )
        sgd_results = sgd_stats
        
        print("\n[Final Evaluation] Testing best SGD weights 50 times...")
        final_sgd_avg, final_sgd_best = evaluate_chromosome(sgd_best_chrom, num_games=50)
        
        sgd_best_game_history = get_best_game_history()
        sgd_best_score_val = sgd_best_game_history['final_score'] if sgd_best_game_history else 0
        print(f"SGD Training Best Score: {sgd_best_score_val}")
        print(f"SGD Final 50-Game Eval - Avg: {final_sgd_avg}, Max: {final_sgd_best}")

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
        
        print(f"\n[Final Evaluation] Testing best Hybrid weights {final_chromosome_test_count} times...")
        final_hybrid_avg, final_hybrid_best = evaluate_chromosome(hybrid_data['best_weights'], num_games=final_chromosome_test_count)
        
        hybrid_best_game_history = get_best_game_history()
        hybrid_best_score_val = hybrid_best_game_history['final_score'] if hybrid_best_game_history else 0
        print(f"Hybrid Training Best Score: {hybrid_best_score_val}")
        print(f"Hybrid Final {final_chromosome_test_count}-Game Eval - Avg: {final_hybrid_avg}, Max: {final_hybrid_best}")

    if run_ga or run_sgd or run_hybrid:
        plt.figure(figsize=(12, 6))
        
        if run_ga and ga_results:
            ga_gens, ga_best_scores, ga_avg_scores, ga_best_fitness, _, ga_chrom_hist = ga_results
            plt.plot(ga_gens, ga_avg_scores, label='GA Average Fitness', color='blue', linestyle='-', alpha=0.5)
            plt.plot(ga_gens, ga_best_scores, label='GA All-Time Best Score', color='blue', linestyle='--', linewidth=2)
            plt.plot(ga_gens, ga_best_fitness, label='GA All-Time Best Fitness', color='purple', linestyle='-.', linewidth=2)
            plot_chromosome_distribution(ga_chrom_hist, ga_gens, "GA")
        
        if run_sgd and sgd_results:
            sgd_iters, sgd_best_scores, sgd_avg_scores, sgd_chrom_hist = sgd_results
            plt.plot(sgd_iters, sgd_avg_scores, label='SGD Average', color='red', linestyle='-', alpha=0.5)
            plt.plot(sgd_iters, sgd_best_scores, label='SGD Best', color='red', linestyle='--', linewidth=2)
            plot_chromosome_distribution(sgd_chrom_hist, sgd_iters, "SGD")
            
        if run_hybrid and hybrid_results:
            h_ga_gens, h_ga_best, h_ga_avg, h_ga_best_fitness, h_ga_chrom_hist = hybrid_results['ga_stats']
            h_sgd_iters, h_sgd_best, h_sgd_avg, h_sgd_chrom_hist = hybrid_results['sgd_stats']
            
            plt.plot(h_ga_gens, h_ga_avg, label='Hybrid (GA Phase) Avg', color='green', linestyle='-', alpha=0.5)
            plt.plot(h_ga_gens, h_ga_best, label='Hybrid (GA Phase) Best Score', color='green', linestyle='--', linewidth=2)
            plt.plot(h_ga_gens, h_ga_best_fitness, label='Hybrid (GA Phase) Best Fitness', color='darkgreen', linestyle='-.', linewidth=2)
            plt.plot(h_sgd_iters, h_sgd_avg, label='Hybrid (SGD Phase) Avg', color='lime', linestyle='-', alpha=0.5)
            plt.plot(h_sgd_iters, h_sgd_best, label='Hybrid (SGD Phase) Best Score', color='lime', linestyle='--', linewidth=2)
            
            # For hybrid, plot the combined history
            combined_gens = h_ga_gens + h_sgd_iters
            combined_chrom_hist = h_ga_chrom_hist + h_sgd_chrom_hist
            plot_chromosome_distribution(combined_chrom_hist, combined_gens, "Hybrid (GA + SGD)")

        plt.figure(1) # Bring focus back to the performance plot
        plt.xlabel('Iteration / Generation')
        plt.ylabel('Score')
        plt.title('Performance Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    print("\n--- Final Results ---")
    if run_ga and ga_best_ind:
        print("Genetic Algorithm Training Best Score:", ga_best_score_val)
        print("Genetic Algorithm Final 50-Game Eval (Score-best) - Avg:", final_ga_avg, "Max:", final_ga_best)
        print("GA Weights (Score-best):", [f"{w:.3f}" for w in ga_best_ind.chromosome])
        print("Genetic Algorithm Final 50-Game Eval (Fitness-best) - Avg:", final_ga_fit_avg, "Max:", final_ga_fit_best)
        print("GA Weights (Fitness-best):", [f"{w:.3f}" for w in ga_best_fitness_ind.chromosome])
        print("-" * 30)
    
    if run_sgd and sgd_best_chrom:
        print("SGD Training Best Score:", sgd_best_score_val)
        print("SGD Final 50-Game Eval - Avg:", final_sgd_avg, "Max:", final_sgd_best)
        print("SGD Weights:", [f"{w:.3f}" for w in sgd_best_chrom])
        print("-" * 30)

    if run_hybrid and hybrid_results:
        print("Hybrid Training Best Score:", hybrid_best_score_val)
        print("Hybrid Final 50-Game Eval - Avg:", final_hybrid_avg, "Max:", final_hybrid_best)
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
