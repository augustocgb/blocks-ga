from genetic_ai import GeneticAlgorithm
from simulate import evaluate_chromosome
from visualizer import visualize_best_game
import matplotlib.pyplot as plt
import numpy as np

from genetic_ai import GeneticAlgorithm
from gradient_ai import GradientDescentAI
from simulate import evaluate_chromosome
from visualizer import visualize_best_game
import matplotlib.pyplot as plt
import numpy as np

# Balanced parameters for roughly equal computational budget
# GA: 30 pop * 30 gen * 3 games/eval = ~2700 games
# SGD: 50 iter * (10 gradient games + 10 eval games) = ~1000-2000 games depending on settings
# We will set SGD to run for enough iterations to roughly match GA or slightly less.

# GA Settings
POPULATION_SIZE = 30
N_GENERATIONS = 30
CHROMOSOME_LENGTH = 5
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
ELITISM_COUNT = 2

# SGD Settings
SGD_ITERATIONS = 50
SGD_LEARNING_RATE = 0.5  # Normalized gradient step size
SGD_PERTURBATION = 0.1

def run_comparison():
    print(f"--- Starting Comparison: GA vs SGD ---")
    
    # --- Run Genetic Algorithm ---
    print("\n[1/2] Running Genetic Algorithm...")
    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        chromosome_length=CHROMOSOME_LENGTH,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        elitism_count=ELITISM_COUNT
    )

    ga_best_ind, (ga_gens, ga_best_scores, ga_avg_scores, _, _) = ga.run_evolution(evaluate_chromosome, N_GENERATIONS)
    
    # --- Run SGD ---
    print("\n[2/2] Running Stochastic Gradient Descent...")
    sgd = GradientDescentAI(
        chromosome_length=CHROMOSOME_LENGTH,
        learning_rate=SGD_LEARNING_RATE,
        perturbation_size=SGD_PERTURBATION
    )
    
    sgd_best_chrom, (sgd_iters, sgd_best_scores, sgd_avg_scores) = sgd.train(evaluate_chromosome, SGD_ITERATIONS)

    # --- Plotting Comparison ---
    plt.figure(figsize=(12, 6))
    
    # Plot GA
    plt.plot(ga_gens, ga_avg_scores, label='GA Average Score', color='blue', linestyle='-')
    plt.plot(ga_gens, ga_best_scores, label='GA Best Score', color='cyan', linestyle='--', alpha=0.6)
    
    # Plot SGD
    # SGD iterations might not align 1:1 with generations, so we plot on the same x-axis index
    plt.plot(sgd_iters, sgd_avg_scores, label='SGD Average Score', color='red', linestyle='-')
    plt.plot(sgd_iters, sgd_best_scores, label='SGD Best Score', color='orange', linestyle='--', alpha=0.6)

    plt.xlabel('Iteration / Generation')
    plt.ylabel('Score')
    plt.title('Performance Comparison: Genetic Algorithm vs SGD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    print("\n--- Final Results ---")
    print("Genetic Algorithm Best Score:", ga_best_ind.best_score)
    print("GA Weights:", [f"{w:.3f}" for w in ga_best_ind.chromosome])
    print("-" * 30)
    print("SGD Best Score (All Time):", sgd.best_score_all_time)
    print("SGD Weights:", [f"{w:.3f}" for w in sgd_best_chrom])
    print("---")

    # Visualize the winner
    if sgd.best_score_all_time > ga_best_ind.best_score:
        print("Displaying Best Game (SGD)")
        # Temporarily hack the 'best_game_history' in simulate to point to SGD if needed
        # But simulate.py tracks the global best. 
        # If we ran them sequentially, the global best might be from either.
        # Let's just run visualize, which picks up the global best from simulate.py
        visualize_best_game()
    else:
        print("Displaying Best Game (GA)")
        visualize_best_game()

if __name__ == "__main__":
    run_comparison()