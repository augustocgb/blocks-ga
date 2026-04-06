from genetic_ai import GeneticAlgorithm
from gradient_ai import GradientDescentAI
from simulate import evaluate_chromosome, get_best_game_history, reset_best_tracking
import multiprocessing as mp
import time

def _ga_worker(args):
    if len(args) == 5:
        chrom, num_games, seed, viz_slots, queue = args
    else:
        chrom, num_games, seed = args
        viz_slots, queue = None, None
        
    callbacks = []
    if queue is not None and viz_slots is not None:
        for i in range(num_games):
            slot = viz_slots[i] if i < len(viz_slots) else None
            if slot is not None:
                r, c = slot
                def make_cb(row=r, col=c):
                    return lambda g, s, p: queue.put(('update', row, col, g, s, p))
                callbacks.append(make_cb())
            else:
                callbacks.append(None)
    else:
        callbacks = None
        
    avg, best = evaluate_chromosome(chrom, num_games=num_games, seed=seed, render_callbacks=callbacks)
    return avg, best

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
                 sgd_momentum=0.5,
                 visualizer=None,
                 realtime_plotter=None,
                 parallel=False):
        
        self.chromosome_length = chromosome_length
        self.ga_pop_size = ga_pop_size
        self.ga_generations = ga_generations
        self.sgd_iterations = sgd_iterations
        self.games_per_eval_ga = games_per_eval_ga
        self.games_per_eval_sgd = games_per_eval_sgd
        self.games_per_perturbation_sgd = games_per_perturbation_sgd
        
        self.sgd_lr = sgd_lr
        self.sgd_perturbation = sgd_perturbation
        self.sgd_momentum = sgd_momentum
        
        self.visualizer = visualizer
        self.realtime_plotter = realtime_plotter
        self.parallel = parallel
        self.evaluation_history = []

    def run(self):
        print("="*60)
        print("STARTING HYBRID OPTIMIZATION".center(60))
        print("="*60)
        print(f"Phase 1: Genetic Algorithm (Global Search)")
        print(f"  - Population: {self.ga_pop_size}")
        print(f"  - Generations: {self.ga_generations}")
        print(f"  - Games/Eval: {self.games_per_eval_ga}")
        print(f"  - Parallel Evaluation: {self.parallel}")
        
        reset_best_tracking()
        
        ga_games_visualized = 0
        live_chrom_hist = []
        live_gens = []
        live_best_scores = []
        live_avg_scores = []
        live_best_chroms = []
        live_best_avg_chroms = []

        def on_ga_gen_start(gen):
            nonlocal ga_games_visualized
            ga_games_visualized = 0

        def on_ga_gen_end(gen, population):
            if self.realtime_plotter:
                best_score = max(ind.best_score for ind in population)
                avg_score = sum(ind.fitness for ind in population) / len(population)
                best_chrom = max(population, key=lambda ind: ind.best_score).chromosome
                best_avg_chrom = max(population, key=lambda ind: ind.fitness).chromosome

                live_chrom_hist.append([ind.chromosome[:] for ind in population])
                live_gens.append(gen)
                live_best_scores.append(best_score)
                live_avg_scores.append(avg_score)
                live_best_chroms.append(best_chrom[:])
                live_best_avg_chroms.append(best_avg_chrom[:])

                self.realtime_plotter.update_plot({
                    'chromosome_history': live_chrom_hist,
                    'generations': live_gens,
                    'best_scores': live_best_scores,
                    'avg_scores': live_avg_scores,
                    'best_chromosomes': live_best_chroms,
                    'best_avg_chromosomes': live_best_avg_chroms,
                    'all_evaluations': self.evaluation_history
                })

        def evaluate_ga(chrom, num_games=self.games_per_eval_ga, seed=None):
            nonlocal ga_games_visualized
            callbacks = []
            if self.visualizer and not self.visualizer.is_stopped and not self.parallel:
                has_visualized = False
                for i in range(num_games):
                    if not has_visualized and ga_games_visualized < self.visualizer.rows * self.visualizer.cols:
                        r = ga_games_visualized // self.visualizer.cols
                        c = ga_games_visualized % self.visualizer.cols
                        
                        def make_cb(row=r, col=c):
                            return lambda g, s, p: self.visualizer.update_cell(row, col, g, s, p)
                        
                        callbacks.append(make_cb())
                        has_visualized = True
                    else:
                        callbacks.append(None)
                if has_visualized:
                    ga_games_visualized += 1
            else:
                callbacks = None
                
            avg_score, best_score = evaluate_chromosome(chrom, num_games=num_games, seed=seed, render_callbacks=callbacks)
            self.evaluation_history.append({'chromosome': chrom[:], 'score': avg_score})
            return avg_score, best_score

        def evaluate_ga_batch(chromosomes):
            import time
            num_games = self.games_per_eval_ga
            manager = mp.Manager()
            queue = manager.Queue() if self.visualizer else None
            
            tasks = []
            slots_assigned = 0
            total_slots = self.visualizer.rows * self.visualizer.cols if self.visualizer else 0
            
            for c in chromosomes:
                viz_slots = []
                has_visualized = False
                for g in range(num_games):
                    if self.visualizer and not has_visualized and slots_assigned < total_slots:
                        r = slots_assigned // self.visualizer.cols
                        col = slots_assigned % self.visualizer.cols
                        viz_slots.append((r, col))
                        has_visualized = True
                    else:
                        viz_slots.append(None)
                if has_visualized:
                    slots_assigned += 1
                tasks.append((c, num_games, None, viz_slots, queue))

            with mp.Pool() as pool:
                async_result = pool.map_async(_ga_worker, tasks)
                
                if self.visualizer:
                    while not async_result.ready() or not queue.empty():
                        if not queue.empty():
                            msg = queue.get()
                            if msg[0] == 'update':
                                _, r, c, grid, score, pieces = msg
                                self.visualizer.update_cell(r, c, grid, score, pieces)
                        else:
                            if not self.visualizer.is_stopped:
                                self.visualizer.handle_events()
                            time.sleep(0.01)
                else:
                    async_result.wait()
                    
                results = async_result.get()
                
            for c, (avg, best) in zip(chromosomes, results):
                self.evaluation_history.append({'chromosome': c[:], 'score': avg})
            return results

        ga = GeneticAlgorithm(
            population_size=self.ga_pop_size,
            chromosome_length=self.chromosome_length,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elitism_count=2
        )

        eval_func = evaluate_ga_batch if self.parallel else evaluate_ga
        best_ga_ind, best_ga_fitness_ind, (ga_gens, ga_best, ga_avg, ga_best_fitness, _, ga_chrom_hist) = ga.run_evolution(eval_func, self.ga_generations, on_gen_start=on_ga_gen_start, on_gen_end=on_ga_gen_end, parallel=self.parallel)
        
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
        
        sgd_visualizer = None
        if self.visualizer and not self.parallel:
            from visualizer import RealtimeGridVisualizer
            sgd_visualizer = RealtimeGridVisualizer(1, 1, delay_ms=10)
        
        def on_sgd_iter_end(iteration, chrom, avg_score, best_score):
            if self.realtime_plotter:
                actual_iter = len(ga_gens) + iteration
                live_chrom_hist.append([chrom[:]])
                live_gens.append(actual_iter)
                live_best_scores.append(best_score)
                live_avg_scores.append(avg_score)
                live_best_chroms.append(chrom[:])
                live_best_avg_chroms.append(chrom[:])
                
                self.realtime_plotter.update_plot({
                    'chromosome_history': live_chrom_hist,
                    'generations': live_gens,
                    'best_scores': live_best_scores,
                    'avg_scores': live_avg_scores,
                    'best_chromosomes': live_best_chroms,
                    'best_avg_chromosomes': live_best_avg_chroms,
                    'all_evaluations': self.evaluation_history
                })

        def evaluate_sgd(chrom, num_games=self.games_per_eval_sgd, seed=None, is_eval=False):
            callbacks = []
            if sgd_visualizer and not sgd_visualizer.is_stopped and is_eval:
                for i in range(num_games):
                    def make_cb():
                        return lambda g, s, p: sgd_visualizer.update_cell(0, 0, g, s, p)

                    callbacks.append(make_cb())
            else:
                callbacks = None
            avg_score, best_score = evaluate_chromosome(chrom, num_games=num_games, seed=seed, render_callbacks=callbacks)
            self.evaluation_history.append({'chromosome': chrom[:], 'score': avg_score})
            return avg_score, best_score

        sgd = GradientDescentAI(
            chromosome_length=self.chromosome_length,
            learning_rate=self.sgd_lr,
            perturbation_size=self.sgd_perturbation,
            momentum=self.sgd_momentum,
            initial_weights=best_ga_ind.chromosome
        )
        
        best_hybrid_chrom, best_hybrid_avg_chrom, (sgd_iters, sgd_best, sgd_avg, sgd_chrom_hist) = sgd.train(
            evaluate_sgd, 
            self.sgd_iterations,
            num_games_per_eval=self.games_per_eval_sgd,
            num_games_per_perturbation=self.games_per_perturbation_sgd,
            on_iter_end=on_sgd_iter_end
        )

        print("\n" + "="*60)
        print("HYBRID OPTIMIZATION COMPLETE".center(60))
        print("="*60)
        print(f"Final Best Score: {sgd.best_score_all_time}")
        print(f"Final Weights (Score-best): {[f'{w:.3f}' for w in best_hybrid_chrom]}")
        print(f"Final Weights (Fitness-best): {[f'{w:.3f}' for w in best_hybrid_avg_chrom]}")
        
        actual_ga_gens = len(ga_gens)
        sgd_iters_shifted = [i + actual_ga_gens for i in sgd_iters]
        
        return {
            'ga_stats': (ga_gens, ga_best, ga_avg, ga_best_fitness, ga_chrom_hist),
            'sgd_stats': (sgd_iters_shifted, sgd_best, sgd_avg, sgd_chrom_hist),
            'best_score': sgd.best_score_all_time,
            'best_weights': best_hybrid_chrom,
            'best_avg_weights': best_hybrid_avg_chrom,
            'best_ga_ind': best_ga_ind,
            'best_ga_fitness_ind': best_ga_fitness_ind,
            'all_evaluations': self.evaluation_history,
            'all_best_chroms': live_best_chroms,
            'all_best_scores': live_best_scores
        }