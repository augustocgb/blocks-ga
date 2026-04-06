import multiprocessing as mp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import webbrowser
import threading
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import numpy as np

def idw_vectorized(xs, ys, zs, X, Y, power=2):
    grid_shape = X.shape
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    dx = xs[:, None] - X_flat[None, :]
    dy = ys[:, None] - Y_flat[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    # Avoid division by zero
    dist = np.maximum(dist, 1e-10)
    weights = 1.0 / (dist**power)
    
    Z_flat = np.sum(weights * zs[:, None], axis=0) / np.sum(weights, axis=0)
    return Z_flat.reshape(grid_shape)

def _plotter_worker(pipe, n_weights, title):
    state = {
        "html": "<html><body style='font-family: sans-serif; padding: 20px;'>Waiting for first generation to complete...</body></html>",
        "version": 0
    }
    
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass # Suppress logging
            
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                wrapper = f"""<html>
                <head>
                    <title>Live Distribution</title>
                    <script>
                        let currentVersion = {state['version']};
                        
                        function checkVersion() {{
                            fetch('/version')
                                .then(r => r.json())
                                .then(data => {{
                                    if (data.version > currentVersion) {{
                                        currentVersion = data.version;
                                        document.getElementById('plot-frame').src = '/plot';
                                    }}
                                }})
                                .catch(e => console.error(e));
                        }}
                        
                        setInterval(checkVersion, 1000);
                    </script>
                    <style>
                        body {{ margin: 0; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
                        iframe {{ flex-grow: 1; border: none; width: 100%; }}
                    </style>
                </head>
                <body>
                    <iframe id="plot-frame" src="/plot"></iframe>
                </body>
                </html>"""
                self.wfile.write(wrapper.encode('utf-8'))
                
            elif self.path == '/plot':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(state['html'].encode('utf-8'))
                
            elif self.path == '/version':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(json.dumps({"version": state['version']}).encode('utf-8'))
                
    server = HTTPServer(('127.0.0.1', 0), Handler)
    port = server.server_address[1]
    
    def run_server():
        server.serve_forever()
        
    threading.Thread(target=run_server, daemon=True).start()
    webbrowser.open(f"http://127.0.0.1:{port}/")
    
    running = True
    while running:
        if pipe.poll(0.05):
            try:
                msg = pipe.recv()
                if msg == 'QUIT':
                    running = False
                    break
                
                if isinstance(msg, dict):
                    chromosome_history = msg['chromosome_history']
                    generations = msg['generations']
                    best_scores = msg.get('best_scores', [])
                    avg_scores = msg.get('avg_scores', [])
                    best_chroms = msg.get('best_chromosomes', [])
                    best_avg_chroms = msg.get('best_avg_chromosomes', [])
                    all_evaluations = msg.get('all_evaluations', [])
                else:
                    chromosome_history, generations = msg
                    best_scores, avg_scores, best_chroms, best_avg_chroms, all_evaluations = [], [], [], [], []
                
                normalized_history = []
                for gen_pop in chromosome_history:
                    if gen_pop and isinstance(gen_pop[0], (float, int)):
                        normalized_history.append([gen_pop])
                    else:
                        normalized_history.append(gen_pop)
                
                has_scores = len(best_scores) > 0 and len(avg_scores) > 0
                has_3d = len(all_evaluations) > 0 and n_weights >= 3
                total_rows = n_weights + 1 if has_scores else n_weights
                
                specs = []
                for r in range(total_rows):
                    row_specs = [{"type": "xy"}]
                    if has_3d:
                        if r == 0:
                            row_specs.append({"type": "scene", "rowspan": total_rows})
                        else:
                            row_specs.append(None)
                    specs.append(row_specs)
                
                fig = make_subplots(
                    rows=total_rows, cols=2 if has_3d else 1, 
                    specs=specs, shared_xaxes=True, vertical_spacing=0.02,
                    column_widths=[0.6, 0.4] if has_3d else [1.0]
                )
                
                row_offset = 0
                if has_scores:
                    fig.add_trace(go.Scatter(x=generations, y=best_scores, mode='lines+markers', name='Best Score', line=dict(color='red')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=generations, y=avg_scores, mode='lines+markers', name='Avg Score', line=dict(color='blue')), row=1, col=1)
                    fig.update_yaxes(title_text='Score', row=1, col=1)
                    row_offset = 1
                
                for w in range(n_weights):
                    target_row = w + 1 + row_offset
                    for i, gen_pop in enumerate(normalized_history):
                        weight_vals = [chrom[w] for chrom in gen_pop]
                        fig.add_trace(go.Box(
                            y=weight_vals,
                            x=[generations[i]] * len(weight_vals),
                            name=f"Gen {generations[i]}",
                            showlegend=False,
                            marker_color='lightblue'
                        ), row=target_row, col=1)
                        
                        if i < len(best_chroms):
                            fig.add_trace(go.Scatter(
                                x=[generations[i]], y=[best_chroms[i][w]], 
                                mode='markers', marker=dict(symbol='star', size=12, color='gold', line=dict(width=1, color='black')), 
                                showlegend=False, name='Best Chrom'
                            ), row=target_row, col=1)
                        
                        if i < len(best_avg_chroms):
                            fig.add_trace(go.Scatter(
                                x=[generations[i]], y=[best_avg_chroms[i][w]], 
                                mode='markers', marker=dict(symbol='star', size=8, color='orange', line=dict(width=1, color='black')), 
                                showlegend=False, name='Best Avg Chrom'
                            ), row=target_row, col=1)

                    fig.update_yaxes(title_text=f'Weight {w}', row=target_row, col=1)
                
                fig.update_xaxes(title_text='Generation / Iteration', row=total_rows, col=1)
                
                if has_3d:
                    idx1, idx2 = 1, 2
                    xs = np.array([e['chromosome'][idx1] for e in all_evaluations])
                    ys = np.array([e['chromosome'][idx2] for e in all_evaluations])
                    zs = np.array([e['score'] for e in all_evaluations])
                    
                    if len(xs) > 1500:
                        xs, ys, zs = xs[-1500:], ys[-1500:], zs[-1500:]
                        
                    margin_x = max((xs.max() - xs.min()) * 0.1, 0.1)
                    margin_y = max((ys.max() - ys.min()) * 0.1, 0.1)
                    
                    grid_x = np.linspace(xs.min() - margin_x, xs.max() + margin_x, 30)
                    grid_y = np.linspace(ys.min() - margin_y, ys.max() + margin_y, 30)
                    X, Y = np.meshgrid(grid_x, grid_y)
                    
                    Z = idw_vectorized(xs, ys, zs, X, Y, power=3)
                    
                    fig.add_trace(go.Surface(
                        x=grid_x, y=grid_y, z=Z, colorscale='Viridis', opacity=0.8, showscale=False, name='Landscape'
                    ), row=1, col=2)
                    
                    fig.add_trace(go.Scatter3d(
                        x=xs, y=ys, z=zs, mode='markers', marker=dict(size=2, color='black', opacity=0.5), name='Evaluations'
                    ), row=1, col=2)
                    
                    if len(best_chroms) > 0:
                        best_xs = [c[idx1] for c in best_chroms]
                        best_ys = [c[idx2] for c in best_chroms]
                        fig.add_trace(go.Scatter3d(
                            x=best_xs, y=best_ys, z=best_scores, mode='lines+markers',
                            line=dict(color='red', width=4), marker=dict(size=5, color='red'), name='Optimization Path'
                        ), row=1, col=2)
                        
                    fig.update_layout(
                        scene=dict(
                            xaxis_title="Weight 1 (Agg Height)",
                            yaxis_title="Weight 2 (Bumpiness)",
                            zaxis_title="Fitness Score",
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                        )
                    )

                fig.update_layout(title_text=f'Training Progress: {title}', height=250 * max(1, total_rows), margin=dict(l=20, r=20, t=40, b=20), showlegend=has_scores)
                
                html = fig.to_html(include_plotlyjs='cdn', full_html=True)
                state['html'] = html
                state['version'] += 1
                
            except EOFError:
                running = False
                break
        time.sleep(0.05)

class RealtimePlotter:
    def __init__(self, n_weights, title):
        self.parent_pipe, child_pipe = mp.Pipe()
        self.process = mp.Process(target=_plotter_worker, args=(child_pipe, n_weights, title))
        self.process.daemon = True
        self.process.start()

    def update_plot(self, payload):
        self.parent_pipe.send(payload)
        
    def close(self):
        self.parent_pipe.send('QUIT')
        self.process.join(timeout=1)

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
    fig = make_subplots(rows=n_weights, cols=1, shared_xaxes=True)
    
    for w in range(n_weights):
        for i, gen_pop in enumerate(chromosome_history):
            weight_vals = [chrom[w] for chrom in gen_pop]
            fig.add_trace(go.Box(
                y=weight_vals,
                x=[generations[i]] * len(weight_vals),
                name=f"Gen {generations[i]}",
                showlegend=False,
                marker_color='blue'
            ), row=w+1, col=1)
        
        fig.update_yaxes(title_text=f'Weight {w}', row=w+1, col=1)
    
    fig.update_xaxes(title_text='Generation / Iteration', row=n_weights, col=1)
    fig.update_layout(title_text=f'Chromosome Distribution Over Time: {title}', height=200 * max(1, n_weights))
    fig.show()

def plot_data_driven_valley(all_evaluations, best_chroms, best_scores):
    if not all_evaluations or len(all_evaluations[0]['chromosome']) < 3:
        print("No evaluation data to plot or chromosome too small.")
        return
        
    idx1, idx2 = 1, 2
    xs = np.array([e['chromosome'][idx1] for e in all_evaluations])
    ys = np.array([e['chromosome'][idx2] for e in all_evaluations])
    zs = np.array([e['score'] for e in all_evaluations])
    
    margin_x = max((xs.max() - xs.min()) * 0.1, 0.1)
    margin_y = max((ys.max() - ys.min()) * 0.1, 0.1)
    
    grid_x = np.linspace(xs.min() - margin_x, xs.max() + margin_x, 50)
    grid_y = np.linspace(ys.min() - margin_y, ys.max() + margin_y, 50)
    X, Y = np.meshgrid(grid_x, grid_y)
    
    Z = idw_vectorized(xs, ys, zs, X, Y, power=3)
    
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=grid_x, y=grid_y, z=Z,
        colorscale='Viridis', opacity=0.8,
        showscale=False,
        name='Fitness Landscape'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=3, color='black', opacity=0.5),
        name='Evaluated Points'
    ))
    
    if len(best_chroms) > 0:
        best_xs = [c[idx1] for c in best_chroms]
        best_ys = [c[idx2] for c in best_chroms]
        best_zs = best_scores
        
        fig.add_trace(go.Scatter3d(
            x=best_xs, y=best_ys, z=best_zs,
            mode='lines+markers',
            line=dict(color='red', width=5),
            marker=dict(size=6, color='red'),
            name='Optimization Path'
        ))
        
    fig.update_layout(
        title="Data-Driven Fitness Landscape (Weight 1 vs Weight 2)",
        scene=dict(
            xaxis_title="Weight 1 (Agg Height)",
            yaxis_title="Weight 2 (Bumpiness)",
            zaxis_title="Fitness Score",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    fig.show()