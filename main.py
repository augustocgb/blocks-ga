from genetic_ai import GeneticAlgorithm
from simulate import evaluate_chromosome
from visualizer import visualize_best_game
import matplotlib.pyplot as plt
import numpy as np

POPULATION_SIZE = 50
N_GENERATIONS = 100
CHROMOSOME_LENGTH = 5
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
ELITISM_COUNT = 2

def run_evolution():
    # Initialize genetic algorithm
    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        chromosome_length=CHROMOSOME_LENGTH,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        elitism_count=ELITISM_COUNT
    )

    # Track best scores per generation
    best_scores = []
    avg_scores = []
    best_fitness = []
    avg_fitness = []
    generations = []
    best_individual, (generations, best_scores, avg_scores, best_fitness, avg_fitness) = ga.run_evolution(evaluate_chromosome, N_GENERATIONS)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_scores, label='Best Score', color='blue')
    plt.plot(generations, avg_scores, label='Average Score', color='green', alpha=0.5)
    plt.plot(generations, best_fitness, label='Best Fitness', color='red')
    plt.plot(generations, avg_fitness, label='Average Fitness', color='yellow', alpha=0.5)
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('Evolution Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nEvolution completed!")
    print(f"Best chromosome found: {best_individual.chromosome}")
    print(f"Final fitness score: {best_individual.fitness}")
    print(f"Best score achieved: {best_individual.best_score}")

    print("---")

    # See all moves from the highest scoring game
    visualize_best_game()

if __name__ == "__main__":
    run_evolution()