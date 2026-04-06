import math
import argparse
import sys
import multiprocessing as mp
from genetic_ai import GeneticAlgorithm
from gradient_ai import GradientDescentAI
from hybrid_optimizer import HybridOptimizer
from simulate import evaluate_chromosome, get_best_game_history, reset_best_tracking
from visualizer import visualize_best_game
from plotter import RealtimePlotter, plot_chromosome_distribution, plot_data_driven_valley
import plotly.graph_objects as go

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

VISUALIZE_GRID_COLS = 5
VISUALIZE_GRID_ROWS = 5

def _ga_worker(args):
    if len(args) == 5:
        chrom, num_games, seed, slot_queue, msg_queue = args
    else:
        chrom, num_games, seed = args
        slot_queue, msg_queue = None, None
        
    avg_score = 0
    best_score = -float('inf')
    
    import random
    import queue
    from simulate import simulate_game, EMPTY_CELL_COLOR, GRID_SIZE
    
    empty_grid = [[EMPTY_CELL_COLOR for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    
    for game_num in range(num_games):
        if seed is not None:
            random.seed(seed + game_num)
            
        my_slot = None
        if slot_queue is not None:
            try:
                my_slot = slot_queue.get_nowait()
            except queue.Empty:
                my_slot = None
                
        cb = None
        if my_slot is not None:
            r, c = my_slot
            def make_cb(row=r, col=c):
                return lambda g, s, p: msg_queue.put(('update', row, col, g, s, p))
            cb = make_cb()
            
        score = simulate_game(chrom, render_callback=cb)
        
        if my_slot is not None:
            r, c = my_slot
            msg_queue.put(('update', r, c, empty_grid, 0, None))
            slot_queue.put(my_slot)
            
        if score > best_score:
            best_score = score
        avg_score += score
        
    if seed is not None:
        random.seed()
        
    return int(avg_score / num_games), best_score

def run_comparison():
    parser = argparse.ArgumentParser(description="Run Block Game Optimization Comparison")
    parser.add_argument('--run-ga', action='store_true', help="Run Genetic Algorithm")
    parser.add_argument('--run-sgd', action='store_true', help="Run Stochastic Gradient Descent")
    parser.add_argument('--run-hybrid', action='store_true', help="Run Hybrid Optimization (GA -> SGD)")
    parser.add_argument('-v', '--visualize', action='store_true', help="Visualize training in real-time")
    parser.add_argument('-p', '--parallel', action='store_true', help="Run evaluation in parallel using multiprocessing")
    
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
        
        ga_evaluation_history = []
        ga_games_visualized = 0
        live_ga_chrom_hist = []
        live_ga_gens = []
        live_ga_best_scores = []
        live_ga_avg_scores = []
        live_ga_best_chroms = []
        live_ga_best_avg_chroms = []

        def on_ga_gen_start(gen):
            nonlocal ga_games_visualized
            ga_games_visualized = 0
            
        def on_ga_gen_end(gen, population):
            if realtime_plotter:
                best_score = max(ind.best_score for ind in population)
                avg_score = sum(ind.fitness for ind in population) / len(population)
                best_chrom = max(population, key=lambda ind: ind.best_score).chromosome
                best_avg_chrom = max(population, key=lambda ind: ind.fitness).chromosome
                
                live_ga_chrom_hist.append([ind.chromosome[:] for ind in population])
                live_ga_gens.append(gen)
                live_ga_best_scores.append(best_score)
                live_ga_avg_scores.append(avg_score)
                live_ga_best_chroms.append(best_chrom[:])
                live_ga_best_avg_chroms.append(best_avg_chrom[:])
                
                realtime_plotter.update_plot({
                    'chromosome_history': live_ga_chrom_hist,
                    'generations': live_ga_gens,
                    'best_scores': live_ga_best_scores,
                    'avg_scores': live_ga_avg_scores,
                    'best_chromosomes': live_ga_best_chroms,
                    'best_avg_chromosomes': live_ga_best_avg_chroms,
                    'all_evaluations': ga_evaluation_history
                })
        
        def evaluate_ga(chrom, num_games=GAMES_PER_EVAL_GA, seed=None):
            nonlocal ga_games_visualized
            callbacks = []
            if visualizer and not visualizer.is_stopped and not args.parallel:
                has_visualized = False
                for i in range(num_games):
                    if not has_visualized and ga_games_visualized < VISUALIZE_GRID_ROWS * VISUALIZE_GRID_COLS:
                        r = ga_games_visualized // VISUALIZE_GRID_COLS
                        c = ga_games_visualized % VISUALIZE_GRID_COLS
                        
                        def make_cb(row=r, col=c):
                            return lambda g, s, p: visualizer.update_cell(row, col, g, s, p)
                        
                        callbacks.append(make_cb())
                        has_visualized = True
                    else:
                        callbacks.append(None)
                if has_visualized:
                    ga_games_visualized += 1
            else:
                callbacks = None
                
            avg_score, best = evaluate_chromosome(chrom, num_games=num_games, seed=seed, render_callbacks=callbacks)
            ga_evaluation_history.append({'chromosome': chrom[:], 'score': avg_score})
            return avg_score, best

        def evaluate_ga_batch(chromosomes):
            import time
            num_games = GAMES_PER_EVAL_GA
            manager = mp.Manager()
            msg_queue = manager.Queue() if visualizer else None
            slot_queue = manager.Queue() if visualizer else None
            
            if visualizer:
                for r in range(VISUALIZE_GRID_ROWS):
                    for c in range(VISUALIZE_GRID_COLS):
                        slot_queue.put((r, c))
            
            tasks = []
            for c in chromosomes:
                tasks.append((c, num_games, None, slot_queue, msg_queue))

            with mp.Pool() as pool:
                async_result = pool.map_async(_ga_worker, tasks)
                
                if visualizer:
                    while not async_result.ready() or not msg_queue.empty():
                        if not msg_queue.empty():
                            msg = msg_queue.get()
                            if msg[0] == 'update':
                                _, r, c, grid, score, pieces = msg
                                visualizer.update_cell(r, c, grid, score, pieces)
                        else:
                            if not visualizer.is_stopped:
                                visualizer.handle_events()
                            time.sleep(0.01)
                else:
                    async_result.wait()
                    
                results = async_result.get()
                
            for c, (avg, best) in zip(chromosomes, results):
                ga_evaluation_history.append({'chromosome': c[:], 'score': avg})
            return results

        ga = GeneticAlgorithm(
            population_size=GA_POPULATION_SIZE,
            chromosome_length=CHROMOSOME_LENGTH,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE,
            elitism_count=ELITISM_COUNT
        )

        eval_func = evaluate_ga_batch if args.parallel else evaluate_ga
        ga_best_ind, ga_best_fitness_ind, ga_stats = ga.run_evolution(eval_func, GA_N_GENERATIONS, on_gen_start=on_ga_gen_start, on_gen_end=on_ga_gen_end, parallel=args.parallel)
        ga_results = ga_stats
        
        ga_best_game_history = get_best_game_history()
        ga_best_score_val = ga_best_game_history['final_score'] if ga_best_game_history else 0
        print(f"GA Training Best Score: {ga_best_score_val}")

    if run_sgd:
        print("\n[SGD] Running Stochastic Gradient Descent...")
        reset_best_tracking()
        
        sgd_evaluation_history = []
        live_sgd_chrom_hist = []
        live_sgd_gens = []
        
        live_sgd_best_scores = []
        live_sgd_avg_scores = []

        sgd_visualizer = None
        if args.visualize and not args.parallel:
            from visualizer import RealtimeGridVisualizer
            sgd_visualizer = RealtimeGridVisualizer(1, 1, delay_ms=10)

        def on_sgd_iter_end(iteration, chrom, avg_score, best_score):
            if realtime_plotter:
                live_sgd_chrom_hist.append([chrom[:]])
                live_sgd_gens.append(iteration)
                live_sgd_best_scores.append(best_score)
                live_sgd_avg_scores.append(avg_score)
                realtime_plotter.update_plot({
                    'chromosome_history': live_sgd_chrom_hist,
                    'generations': live_sgd_gens,
                    'best_scores': live_sgd_best_scores,
                    'avg_scores': live_sgd_avg_scores,
                    'best_chromosomes': [chrom[:]] * len(live_sgd_gens),
                    'best_avg_chromosomes': [chrom[:]] * len(live_sgd_gens),
                    'all_evaluations': sgd_evaluation_history
                })
        
        def evaluate_sgd(chrom, num_games=GAMES_PER_EVAL_SGD, seed=None, is_eval=False):
            callbacks = []
            if sgd_visualizer and not sgd_visualizer.is_stopped and is_eval:
                for i in range(num_games):
                    def make_cb():
                        return lambda g, s, p: sgd_visualizer.update_cell(0, 0, g, s, p)
                    
                    callbacks.append(make_cb())
            else:
                callbacks = None
                
            avg_score, best = evaluate_chromosome(chrom, num_games=num_games, seed=seed, render_callbacks=callbacks)
            sgd_evaluation_history.append({'chromosome': chrom[:], 'score': avg_score})
            return avg_score, best

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
            realtime_plotter=realtime_plotter,
            parallel=args.parallel
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
        fig = go.Figure()
        
        if run_ga and ga_results:
            ga_gens, ga_best_scores, ga_avg_scores, ga_best_fitness, _, ga_chrom_hist = ga_results
            fig.add_trace(go.Scatter(x=ga_gens, y=ga_avg_scores, mode='lines', name='GA Average Fitness', line=dict(color='rgba(0,0,255,0.5)', dash='solid')))
            fig.add_trace(go.Scatter(x=ga_gens, y=ga_best_scores, mode='lines', name='GA All-Time Best Score', line=dict(color='blue', dash='dash', width=2)))
            fig.add_trace(go.Scatter(x=ga_gens, y=ga_best_fitness, mode='lines', name='GA All-Time Best Fitness', line=dict(color='purple', dash='dashdot', width=2)))
            plot_chromosome_distribution(ga_chrom_hist, ga_gens, "GA")
        
        if run_sgd and sgd_results:
            sgd_iters, sgd_best_scores, sgd_avg_scores, sgd_chrom_hist = sgd_results
            fig.add_trace(go.Scatter(x=sgd_iters, y=sgd_avg_scores, mode='lines', name='SGD Average', line=dict(color='rgba(255,0,0,0.5)', dash='solid')))
            fig.add_trace(go.Scatter(x=sgd_iters, y=sgd_best_scores, mode='lines', name='SGD Best', line=dict(color='red', dash='dash', width=2)))
            plot_chromosome_distribution(sgd_chrom_hist, sgd_iters, "SGD")
            
        if run_hybrid and hybrid_results:
            h_ga_gens, h_ga_best, h_ga_avg, h_ga_best_fitness, h_ga_chrom_hist = hybrid_results['ga_stats']
            h_sgd_iters, h_sgd_best, h_sgd_avg, h_sgd_chrom_hist = hybrid_results['sgd_stats']
            
            fig.add_trace(go.Scatter(x=h_ga_gens, y=h_ga_avg, mode='lines', name='Hybrid (GA Phase) Avg', line=dict(color='rgba(0,128,0,0.5)', dash='solid')))
            fig.add_trace(go.Scatter(x=h_ga_gens, y=h_ga_best, mode='lines', name='Hybrid (GA Phase) Best Score', line=dict(color='green', dash='dash', width=2)))
            fig.add_trace(go.Scatter(x=h_ga_gens, y=h_ga_best_fitness, mode='lines', name='Hybrid (GA Phase) Best Fitness', line=dict(color='darkgreen', dash='dashdot', width=2)))
            fig.add_trace(go.Scatter(x=h_sgd_iters, y=h_sgd_avg, mode='lines', name='Hybrid (SGD Phase) Avg', line=dict(color='rgba(0,255,0,0.5)', dash='solid')))
            fig.add_trace(go.Scatter(x=h_sgd_iters, y=h_sgd_best, mode='lines', name='Hybrid (SGD Phase) Best Score', line=dict(color='lime', dash='dash', width=2)))
            
            # For hybrid, plot the combined history
            combined_gens = h_ga_gens + h_sgd_iters
            combined_chrom_hist = h_ga_chrom_hist + h_sgd_chrom_hist
            plot_chromosome_distribution(combined_chrom_hist, combined_gens, "Hybrid (GA + SGD)")

        fig.update_layout(
            title='Performance Comparison',
            xaxis_title='Iteration / Generation',
            yaxis_title='Score',
            template='plotly_white'
        )
        fig.show()

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
        
        print("\nGenerating Data-Driven GA vs SGD Fitness Landscape Visual...")
        plot_data_driven_valley(hybrid_results['all_evaluations'], hybrid_results['all_best_chroms'], hybrid_results['all_best_scores'])

if __name__ == "__main__":
    run_comparison()