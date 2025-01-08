import numpy as np
from scipy.integrate import odeint
import networkx as nx
import heapq
import matplotlib.pyplot as plt
import time

# Fungsi model dan fungsi error sama seperti sebelumnya
def lotka_volterra(state, t, alpha, beta, delta, gamma):
    prey, predator = state
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    return [dprey_dt, dpredator_dt]

def calculate_error(params, data, t):
    alpha, beta, delta, gamma = params
    initial_state = [data[0, 0], data[0, 1]]
    solution = odeint(lotka_volterra, initial_state, t, args=(alpha, beta, delta, gamma))
    return np.mean((solution - data)**2)

# Generate data dengan ukuran berbeda
def generate_data(n_points, noise_level=0.1):
    true_params = {
        'alpha': 0.8,
        'beta': 0.08,
        'delta': 0.07,
        'gamma': 0.45
    }
    
    t = np.linspace(0, 100, n_points)
    initial_state = [10.0, 5.0]
    
    solution = odeint(lotka_volterra, initial_state, t, 
                     args=(true_params['alpha'], true_params['beta'], 
                           true_params['delta'], true_params['gamma']))
    
    noise = np.random.normal(0, noise_level, solution.shape)
    noisy_data = solution + noise
    noisy_data = np.maximum(noisy_data, 0)
    
    return t, noisy_data, true_params

# Membuat graf dengan kompleksitas berbeda
def create_parameter_graph(data, t, complexity=5):
    G = nx.Graph()
    param_ranges = {
        'alpha': np.linspace(0.5, 1.5, complexity),
        'beta': np.linspace(0.05, 0.15, complexity),
        'delta': np.linspace(0.05, 0.1, complexity),
        'gamma': np.linspace(0.3, 0.7, complexity)
    }
    
    param_combinations = []
    for alpha in param_ranges['alpha']:
        for beta in param_ranges['beta']:
            for delta in param_ranges['delta']:
                for gamma in param_ranges['gamma']:
                    params = (alpha, beta, delta, gamma)
                    param_combinations.append(params)
    
    for i, params in enumerate(param_combinations):
        error = calculate_error(params, data, t)
        G.add_node(i, params=params, error=error)
    
    for i in range(len(param_combinations)):
        for j in range(i+1, len(param_combinations)):
            diff = np.sum(np.abs(np.array(param_combinations[i]) - np.array(param_combinations[j])))
            if diff < 0.1:
                weight = abs(G.nodes[i]['error'] - G.nodes[j]['error'])
                G.add_edge(i, j, weight=weight)
    
    return G, param_combinations

# Optimasi Dijkstra
def dijkstra_optimization(G, param_combinations):
    start_time = time.perf_counter()
    
    start_node = 0
    distances = {node: float('infinity') for node in G.nodes()}
    distances[start_node] = G.nodes[start_node]['error']
    pq = [(G.nodes[start_node]['error'], start_node)]
    best_node = start_node
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_distance > distances[current_node]:
            continue
            
        if G.nodes[current_node]['error'] < G.nodes[best_node]['error']:
            best_node = current_node
            
        for neighbor in G.neighbors(current_node):
            distance = current_distance + G[current_node][neighbor]['weight']
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    end_time = time.perf_counter()
    return param_combinations[best_node], G.nodes[best_node]['error'], end_time - start_time

# Optimasi DFS
def dfs_optimization(G, param_combinations):
    start_time = time.perf_counter()
    
    visited = set()
    best_params = None
    best_error = float('infinity')
    
    def dfs_recursive(node):
        nonlocal best_params, best_error
        visited.add(node)
        
        current_error = G.nodes[node]['error']
        if current_error < best_error:
            best_error = current_error
            best_params = param_combinations[node]
            
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                dfs_recursive(neighbor)
    
    dfs_recursive(0)
    end_time = time.perf_counter()
    return best_params, best_error, end_time - start_time

def main():
    # Ukuran data yang akan diuji
    data_sizes = [100, 500, 1000, 2000, 5000]
    # Kompleksitas graf (jumlah titik per parameter)
    graph_complexities = [3, 4, 5, 6, 7]
    
    print("\nAnalisis Performa dengan Berbagai Ukuran Data dan Kompleksitas Graf:")
    print("\nUkuran Data | Kompleksitas | Nodes | Edges | Waktu Dijkstra | Waktu DFS | Error Dijkstra | Error DFS")
    print("-" * 100)
    
    for n in data_sizes:
        for complexity in graph_complexities:
            # Generate data
            t, data, true_params = generate_data(n)
            
            # Buat graf
            G, param_combinations = create_parameter_graph(data, t, complexity)
            
            # Optimasi dengan Dijkstra
            best_params_dijkstra, error_dijkstra, time_dijkstra = dijkstra_optimization(G, param_combinations)
            
            # Optimasi dengan DFS
            best_params_dfs, error_dfs, time_dfs = dfs_optimization(G, param_combinations)
            
            print(f"{n:^11d} | {complexity:^11d} | {G.number_of_nodes():^6d} | {G.number_of_edges():^6d} | "
                  f"{time_dijkstra:^14.6f} | {time_dfs:^9.6f} | {error_dijkstra:^14.6f} | {error_dfs:^9.6f}")
            
            # Demonstrasi multiple runs untuk ukuran dan kompleksitas terbesar
            if n == data_sizes[-1] and complexity == graph_complexities[-1]:
                print(f"\nDemonstrasi Multiple Runs untuk n = {n}, kompleksitas = {complexity}:")
                num_runs = 5
                
                # Multiple runs untuk Dijkstra
                start_time = time.perf_counter()
                for _ in range(num_runs):
                    _, _, _ = dijkstra_optimization(G, param_combinations)
                dijkstra_multi_time = time.perf_counter() - start_time
                
                # Multiple runs untuk DFS
                start_time = time.perf_counter()
                for _ in range(num_runs):
                    _, _, _ = dfs_optimization(G, param_combinations)
                dfs_multi_time = time.perf_counter() - start_time
                
                print(f"\nWaktu untuk {num_runs} runs:")
                print(f"Dijkstra: {dijkstra_multi_time:.6f} s")
                print(f"DFS     : {dfs_multi_time:.6f} s")
    
    # Plot contoh hasil untuk ukuran data menengah
    mid_size = data_sizes[len(data_sizes)//2]
    t, data, _ = generate_data(mid_size)
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, data[:, 0], 'b.', label='Data Mangsa', alpha=0.5)
    plt.plot(t, data[:, 1], 'r.', label='Data Predator', alpha=0.5)
    plt.title(f'Data Example (n={mid_size})')
    plt.xlabel('Waktu')
    plt.ylabel('Populasi')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()