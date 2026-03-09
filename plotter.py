import multiprocessing as mp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import webbrowser
import threading
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import os

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
                
                # Assume msg is a dictionary payload if it's a dict, otherwise fallback to tuple
                if isinstance(msg, dict):
                    chromosome_history = msg['chromosome_history']
                    generations = msg['generations']
                    best_scores = msg.get('best_scores', [])
                    avg_scores = msg.get('avg_scores', [])
                    best_chroms = msg.get('best_chromosomes', [])
                    best_avg_chroms = msg.get('best_avg_chromosomes', [])
                else:
                    chromosome_history, generations = msg
                    best_scores = []
                    avg_scores = []
                    best_chroms = []
                    best_avg_chroms = []
                
                normalized_history = []
                for gen_pop in chromosome_history:
                    if gen_pop and isinstance(gen_pop[0], (float, int)):
                        normalized_history.append([gen_pop])
                    else:
                        normalized_history.append(gen_pop)
                
                has_scores = len(best_scores) > 0 and len(avg_scores) > 0
                total_rows = n_weights + 1 if has_scores else n_weights
                
                fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.02)
                
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
                                x=[generations[i]], 
                                y=[best_chroms[i][w]], 
                                mode='markers', 
                                marker=dict(symbol='star', size=12, color='gold', line=dict(width=1, color='black')), 
                                showlegend=False, 
                                name='Best Chrom'
                            ), row=target_row, col=1)
                        
                        if i < len(best_avg_chroms):
                            fig.add_trace(go.Scatter(
                                x=[generations[i]], 
                                y=[best_avg_chroms[i][w]], 
                                mode='markers', 
                                marker=dict(symbol='star', size=8, color='orange', line=dict(width=1, color='black')), 
                                showlegend=False, 
                                name='Best Avg Chrom'
                            ), row=target_row, col=1)

                    fig.update_yaxes(title_text=f'Weight {w}', row=target_row, col=1)
                
                fig.update_xaxes(title_text='Generation / Iteration', row=total_rows, col=1)
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
