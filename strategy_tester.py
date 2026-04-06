import multiprocessing as mp
import plotly.graph_objects as go
from simulate import evaluate_chromosome

# ==============================================================================
# STRATEGY CONFIGURATION
# ==============================================================================

# These are just example strategies I've gotten from a round of testing:
STRATEGIES = {
    "Hybrid Strategy": [  0.315,  -0.065,   0.092,   0.988,   0.867,   0.902],
    "Pure GA Strategy": [-0.290, 0.025, -0.115, 1.202, 0.754, 0.724],
    "Pure SGD Strategy": [ -0.046,   0.188,  -0.113,   0.027,   0.077,   0.155]
}

# Number of games to play per strategy to calculate the average fitness and best score
NUM_GAMES_PER_STRATEGY = 100

def _evaluate_strategy(args):
    name, weights, num_games = args
    print(f"Running '{name}' for {num_games} games...")
    avg_score, best_score = evaluate_chromosome(weights, num_games=num_games)
    print(f"  -> [{name}] Avg: {avg_score} | Max: {best_score}")
    return name, avg_score, best_score

def run_tests():
    print("=" * 60)
    print(f"STRATEGY TESTER ({NUM_GAMES_PER_STRATEGY} Games per Strategy)")
    print("=" * 60)
    
    tasks = [(name, weights, NUM_GAMES_PER_STRATEGY) for name, weights in STRATEGIES.items()]
    
    # Run games in parallel to save time
    with mp.Pool() as pool:
        results = pool.map(_evaluate_strategy, tasks)
        
    names = [r[0] for r in results]
    avg_scores = [r[1] for r in results]
    best_scores = [r[2] for r in results]

    print("\n" + "=" * 60)
    print("TESTING COMPLETE. GENERATING PLOT...")
    print("=" * 60)

    # Generate the visualization
    fig = go.Figure()
    
    # Average Score (Fitness) Bars
    fig.add_trace(go.Bar(
        x=names,
        y=avg_scores,
        name='Average Score (Fitness)',
        marker_color='rgba(55, 128, 191, 0.8)',
        text=avg_scores,
        textposition='auto',
        hoverinfo='y'
    ))
    
    # Highest Score Bars
    fig.add_trace(go.Bar(
        x=names,
        y=best_scores,
        name='Highest Single-Game Score',
        marker_color='rgba(219, 64, 82, 0.8)',
        text=best_scores,
        textposition='auto',
        hoverinfo='y'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'Strategy Performance Comparison ({NUM_GAMES_PER_STRATEGY} Games per Strategy)',
            font=dict(size=24)
        ),
        xaxis_title='Strategy Name',
        yaxis_title='Game Score',
        barmode='group',
        template='plotly_white',
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Make the plot look good for a poster
    fig.update_xaxes(tickfont=dict(size=14))
    fig.update_yaxes(tickfont=dict(size=14))
    
    fig.show()

if __name__ == "__main__":
    # Protects the entry point for multiprocessing in Windows
    run_tests()