import argparse
import sys
import multiprocessing as mp
from genetic_ai import GeneticAlgorithm
from gradient_ai import GradientDescentAI
from hybrid_optimizer import HybridOptimizer
from simulate import evaluate_chromosome, get_best_game_history, reset_best_tracking
from visualizer import visualize_best_game
import matplotlib.pyplot as plt
import numpy as np

def _plotter_worker(pipe, n_weights, title):
    import matplotlib.pyplot as plt
    plt.ion()
    fig, axes = plt.subplots(n_weights, 1, figsize=(10, 2 * n_weights), sharex=True)
    if n_weights == 1: axes = [axes]
    
    fig.suptitle(f'Chromosome Distribution Over Time: {title}')
    axes[-1].set_xlabel('Generation / Iteration')
    
    for w in range(n_weights):
        axes[w].set_ylabel(f'Weight {w}')
        axes[w].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show(block=False)
    
    running = True
    while running:
        if pipe.poll(0.05):
            try:
                msg = pipe.recv()
                if msg == 'QUIT':
                    running = False
                    break
                
                chromosome_history, generations = msg
                
                normalized_history = []
                for gen_pop in chromosome_history:
                    if gen_pop and isinstance(gen_pop[0], (float, int)):
                        normalized_history.append([gen_pop])
                    else:
                        normalized_history.append(gen_pop)
                
                for w in range(n_weights):
                    axes[w].clear()
                    axes[w].set_ylabel(f'Weight {w}')
                    axes[w].grid(True, alpha=0.3)
                    
                    data = []
                    for gen_pop in normalized_history:
                        weight_vals = [chrom[w] for chrom in gen_pop]
                        data.append(weight_vals)
                    
                    axes[w].boxplot(data, positions=generations, widths=0.6, manage_ticks=False)
                    
                axes[-1].set_xlabel('Generation / Iteration')
                if len(generations) > 1:
                    axes[-1].set_xlim(min(generations) - 1, max(generations) + 1)
            except EOFError:
                running = False
                break
                
        try:
            plt.pause(0.05)
        except Exception:
            running = False
            break

class RealtimePlotter:
    def __init__(self, n_weights, title):
        self.parent_pipe, child_pipe = mp.Pipe()
        self.process = mp.Process(target=_plotter_worker, args=(child_pipe, n_weights, title))
        self.process.daemon = True
        self.process.start()

    def update_plot(self, chromosome_history, generations):
        self.parent_pipe.send((chromosome_history, generations))
        
    def close(self):
        self.parent_pipe.send('QUIT')
        self.process.join(timeout=1)

# --- Configuration ---
GA_POPULATION_SIZE = 100
GA_N_GENERATIONS = 10
GAMES_PER_EVAL_GA = 10

CHROMOSOME_LENGTH = 6
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
ELITISM_COUNT = 2

final_chromosome_test_count = 100

SGD_ITERATIONS = 10
GAMES_PER_PERTURBATION_SGD = 10
GAMES_PER_EVAL_SGD = 50

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

VISUALIZE_GRID_ROWS = 5
VISUALIZE_GRID_COLS = 5

def run_comparison():
    parser = argparse.ArgumentParser(description="Run Block Game Optimization Comparison")
    parser.add_argument('--run-ga', action='store_true', help="Run Genetic Algorithm")
    parser.add_argument('--run-sgd', action='store_true', help="Run Stochastic Gradient Descent")
    parser.add_argument('--run-hybrid', action='store_true', help="Run Hybrid Optimization (GA -> SGD)")
    parser.add_argument('-v', '--visualize', action='store_true', help="Visualize training in real-time")
    
    args = parser.parse_args()

    if not args.run_ga and not args.run_sgd and not args.run_hybrid:
        run_ga, run_sgd, run_hybrid = True, True, False
    else:
        run_ga, run_sgd, run_hybrid = args.run_ga, args.run_sgd, args.run_hybrid

    print(f"--- Starting Optimization Run ---")
    
    visualizer = None
    realtime_plotter = None
    if args.visualize:
        from visualizer import RealtimeGridVisualizer
        visualizer = RealtimeGridVisualizer(VISUALIZE_GRID_ROWS, VISUALIZE_GRID_COLS, delay_ms=10)
        realtime_plotter = RealtimePlotter(CHROMOSOME_LENGTH, "Live Training Distribution")

    ga_results, sgd_results, hybrid_results = None, None, None
    ga_best_ind, ga_best_game_history = None, None
    sgd_best_chrom, sgd_best_game_history, sgd_best_score_val = None, None, 0
    hybrid_best_score_val, hybrid_best_game_history = 0, None

    if run_ga:
        print("\n[GA] Running Genetic Algorithm...")
        reset_best_tracking()
        
        ga_games_visualized = 0
        live_ga_chrom_hist = []
        live_ga_gens = []

        def on_ga_gen_start(gen):
            nonlocal ga_games_visualized
            ga_games_visualized = 0
            
        def on_ga_gen_end(gen, population):
            if realtime_plotter:
                live_ga_chrom_hist.append([ind.chromosome[:] for ind in population])
                live_ga_gens.append(gen)
                realtime_plotter.update_plot(live_ga_chrom_hist, live_ga_gens)
        
        def evaluate_ga(chrom, num_games=GAMES_PER_EVAL_GA, seed=None):
            nonlocal ga_games_visualized
            callbacks = []
            if visualizer and not visualizer.is_stopped:
                for i in range(num_games):
                    idx = ga_games_visualized % (VISUALIZE_GRID_ROWS * VISUALIZE_GRID_COLS)
                    r = idx // VISUALIZE_GRID_COLS
                    c = idx % VISUALIZE_GRID_COLS
                    
                    def make_cb(row, col):
                        return lambda g, s, p: visualizer.update_cell(row, col, g, s, p)
                    
                    callbacks.append(make_cb(r, c))
                    ga_games_visualized += 1
            else:
                callbacks = None
                
            return evaluate_chromosome(chrom, num_games=num_games, seed=seed, render_callbacks=callbacks)

        ga = GeneticAlgorithm(
            population_size=GA_POPULATION_SIZE,
            chromosome_length=CHROMOSOME_LENGTH,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE,
            elitism_count=ELITISM_COUNT
        )

        ga_best_ind, ga_best_fitness_ind, ga_stats = ga.run_evolution(evaluate_ga, GA_N_GENERATIONS, on_gen_start=on_ga_gen_start, on_gen_end=on_ga_gen_end)
        ga_results = ga_stats
        
        ga_best_game_history = get_best_game_history()
        ga_best_score_val = ga_best_game_history['final_score'] if ga_best_game_history else 0
        print(f"GA Training Best Score: {ga_best_score_val}")

    if run_sgd:
        print("\n[SGD] Running Stochastic Gradient Descent...")
        reset_best_tracking()
        
        live_sgd_chrom_hist = []
        live_sgd_gens = []
        
        sgd_visualizer = None
        if args.visualize:
            from visualizer import RealtimeGridVisualizer
            sgd_visualizer = RealtimeGridVisualizer(1, 1, delay_ms=10)

        def on_sgd_iter_end(iteration, chrom):
            if realtime_plotter:
                live_sgd_chrom_hist.append(chrom[:])
                live_sgd_gens.append(iteration)
                realtime_plotter.update_plot(live_sgd_chrom_hist, live_sgd_gens)
        
        def evaluate_sgd(chrom, num_games=GAMES_PER_EVAL_SGD, seed=None, is_eval=False):
            callbacks = []
            if sgd_visualizer and not sgd_visualizer.is_stopped and is_eval:
                for i in range(num_games):
                    def make_cb():
                        return lambda g, s, p: sgd_visualizer.update_cell(0, 0, g, s, p)
                    
                    callbacks.append(make_cb())
            else:
                callbacks = None
                
            return evaluate_chromosome(chrom, num_games=num_games, seed=seed, render_callbacks=callbacks)

        sgd = GradientDescentAI(
            chromosome_length=CHROMOSOME_LENGTH,
            learning_rate=SGD_LEARNING_RATE,
            perturbation_size=SGD_PERTURBATION,
            momentum=SGD_MOMENTUM
        )
        
        sgd_best_chrom, sgd_best_avg_chrom, sgd_stats = sgd.train(
            evaluate_sgd, 
            SGD_ITERATIONS,
            num_games_per_eval=GAMES_PER_EVAL_SGD,
            num_games_per_perturbation=GAMES_PER_PERTURBATION_SGD,
            on_iter_end=on_sgd_iter_end
        )
        sgd_results = sgd_stats
        
        sgd_best_game_history = get_best_game_history()
        sgd_best_score_val = sgd_best_game_history['final_score'] if sgd_best_game_history else 0
        print(f"SGD Training Best Score: {sgd_best_score_val}")

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
            sgd_momentum=SGD_MOMENTUM,
            visualizer=visualizer,
            realtime_plotter=realtime_plotter
        )
        
        hybrid_data = optimizer.run()
        hybrid_results = hybrid_data
        
        hybrid_best_game_history = get_best_game_history()
        hybrid_best_score_val = hybrid_best_game_history['final_score'] if hybrid_best_game_history else 0
        print(f"Hybrid Training Best Score: {hybrid_best_score_val}")

    if realtime_plotter:
        realtime_plotter.close()

    print(f"\n{'='*60}")
    print(f"STARTING FINAL EVALUATIONS ({final_chromosome_test_count} GAMES EACH)".center(60))
    print(f"{'='*60}")
    
    if run_ga:
        print(f"\n[Final Evaluation] Testing best GA weights (Score-based)...")
        final_ga_avg, final_ga_best = evaluate_chromosome(ga_best_ind.chromosome, num_games=final_chromosome_test_count)
        
        print(f"[Final Evaluation] Testing best GA weights (Fitness-based)...")
        final_ga_fit_avg, final_ga_fit_best = evaluate_chromosome(ga_best_fitness_ind.chromosome, num_games=final_chromosome_test_count)
        
    if run_sgd:
        print(f"\n[Final Evaluation] Testing best SGD weights (Score-based)...")
        final_sgd_avg, final_sgd_best = evaluate_chromosome(sgd_best_chrom, num_games=final_chromosome_test_count)
        
        print(f"[Final Evaluation] Testing best SGD weights (Fitness-based)...")
        final_sgd_fit_avg, final_sgd_fit_best = evaluate_chromosome(sgd_best_avg_chrom, num_games=final_chromosome_test_count)
        
    if run_hybrid:
        print(f"\n[Final Evaluation] Testing best Hybrid weights (Score-based)...")
        final_hybrid_avg, final_hybrid_best = evaluate_chromosome(hybrid_data['best_weights'], num_games=final_chromosome_test_count)
        
        print(f"[Final Evaluation] Testing best Hybrid weights (Fitness-based)...")
        final_hybrid_fit_avg, final_hybrid_fit_best = evaluate_chromosome(hybrid_data['best_avg_weights'], num_games=final_chromosome_test_count)

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
        print(f"Genetic Algorithm Final {final_chromosome_test_count}-Game Eval (Score-best) - Avg:", final_ga_avg, "Max:", final_ga_best)
        print("GA Weights (Score-best):", [f"{w:.3f}" for w in ga_best_ind.chromosome])
        print(f"Genetic Algorithm Final {final_chromosome_test_count}-Game Eval (Fitness-best) - Avg:", final_ga_fit_avg, "Max:", final_ga_fit_best)
        print("GA Weights (Fitness-best):", [f"{w:.3f}" for w in ga_best_fitness_ind.chromosome])
        print("-" * 30)
    
    if run_sgd and sgd_best_chrom:
        print("SGD Training Best Score:", sgd_best_score_val)
        print(f"SGD Final {final_chromosome_test_count}-Game Eval (Score-best) - Avg:", final_sgd_avg, "Max:", final_sgd_best)
        print("SGD Weights (Score-best):", [f"{w:.3f}" for w in sgd_best_chrom])
        print(f"SGD Final {final_chromosome_test_count}-Game Eval (Fitness-best) - Avg:", final_sgd_fit_avg, "Max:", final_sgd_fit_best)
        print("SGD Weights (Fitness-best):", [f"{w:.3f}" for w in sgd_best_avg_chrom])
        print("-" * 30)

    if run_hybrid and hybrid_results:
        print("Hybrid Training Best Score:", hybrid_best_score_val)
        print(f"Hybrid Final {final_chromosome_test_count}-Game Eval (Score-best) - Avg:", final_hybrid_avg, "Max:", final_hybrid_best)
        print("Hybrid Weights (Score-best):", [f"{w:.3f}" for w in hybrid_results['best_weights']])
        print(f"Hybrid Final {final_chromosome_test_count}-Game Eval (Fitness-best) - Avg:", final_hybrid_fit_avg, "Max:", final_hybrid_fit_best)
        print("Hybrid Weights (Fitness-best):", [f"{w:.3f}" for w in hybrid_results['best_avg_weights']])
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
