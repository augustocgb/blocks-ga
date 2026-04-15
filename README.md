# Blocks Genetic Algorithm & SGD Optimizer (blocks-ga)

**Note:** This project was presented as a research project for the National Conference on Undergraduate Research (NCUR) 2026.

This project implements and compares various optimization algorithm: Genetic Algorithms (GA), Stochastic Gradient Descent (SGD), and a Hybrid approach to optimize gameplay for a block placement game (similar to Tetris). The AI evolves over time to maximize its score by learning better strategies for placing blocks on a grid through weight tuning of a heuristic function.

## Features
- **Multiple Optimizers**: 
  - **Genetic Algorithm (GA)**: A population-based global search.
  - **Stochastic Gradient Descent (SGD)**: A local search optimization using finite differences for gradient estimation in a non-differentiable environment.
  - **Hybrid Optimizer**: Combines both methods, utilizing GA for initial global exploration and then applying SGD to fine-tune the best weights.
- **Game Simulation**: Simulates gameplay to evaluate agents (chromosomes/weight vectors).
- **Realtime Visualizer**: Replays the best game from the evolution or optimization process in real-time.
- **Customizable Parameters**: Easily adjust population size, mutation rate, learning rate, iterations, and more.

## Heuristic Features
The AI evaluates potential moves based on six structural grid features:
1. **Holes:** Empty spaces trapped beneath blocks.
2. **Aggregate Height:** The sum of the heights of all columns.
3. **Bumpiness:** The variance in height between adjacent columns.
4. **Potential Lines Cleared:** The immediate reward of a move.
5. **Contact Points:** The degree to which a piece fits snugly into the existing structure.
6. **Predicted Streak:** The potential for consecutive line clears.

## Requirements
- Python 3.8+
- Required libraries:
  - `pygame`
  - `plotly`
  - `numpy`

Install the dependencies using:
```bash
pip install pygame plotly numpy
```

## How to Run

### 1. Run the Optimization Process
The main script runs the chosen optimization algorithm (GA, SGD, or Hybrid) and visualizes the results.
```bash
python main.py
```
*(Note: You can customize which optimizer to run by modifying the parameters in `main.py`)*

### 2. Realtime Visualizer
After the optimization process, statistics are plotted and the best game is automatically replayed in the realtime visualizer.

**Controls for Visualizer:**
- **Right Arrow**: Step forward through moves (hold for continuous steps).
- **Left Arrow**: Step backward through moves (hold for continuous steps).
- **Spacebar**: Pause/Unpause the replay.
- **Up Arrow**: Decrease replay delay (speed up).
- **Down Arrow**: Increase replay delay (slow down).
- **Escape**: Exit the visualizer.

### 3. Play the Game Yourself!
Try to beat the best optimized agent.
```bash
python game.py
```

## Project Structure
```text
blocks-ga/
├── game.py              # Core game logic and block shapes
├── genetic_ai.py        # Genetic algorithm implementation
├── gradient_ai.py       # SGD implementation using finite differences
├── hybrid_optimizer.py  # Hybrid (GA + SGD) optimizer implementation
├── simulate.py          # Game simulation for fitness evaluation
├── visualizer.py        # Realtime visualizer for replaying games
├── main.py              # Entry point for running the optimization process
└── README.md            # Project documentation
```

## How It Works (Methodology)
1. **Initialization**: Depending on the algorithm, a population of random chromosomes (GA) or a single weight vector (SGD) is created.
2. **Simulation**: Each agent is evaluated by simulating multiple games. The fitness is the average of the scores.
3. **Optimization**:
   - **GA**: The best-performing agents are selected as parents. New agents are generated via crossover and mutation.
   - **SGD**: Gradients are estimated using central finite differences, and weights are updated using an SGD rule with momentum.
   - **Hybrid**: GA discovers a strong baseline policy, which is then fine-tuned by SGD.
4. **Repeat**: The simulation and optimization steps are repeated for multiple generations or iterations.
5. **Visualization**: Performance improves over time, and the best game is replayed using the realtime visualizer.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.