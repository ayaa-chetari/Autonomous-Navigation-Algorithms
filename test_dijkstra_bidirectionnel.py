import networkx as nx
import random
import time
import heapq
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no Tkinter required)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dijkstra_bidirectionnel import dijkstra_bidirectionnel, dijkstra_classique, astar_bidirectionnel
from Maze import Maze


class GraphAdapter:
    """Adapter for NetworkX graphs to work with dijkstra algorithm"""
    
    def __init__(self, nx_graph):
        self.graph = nx_graph
    
    def get_neighbors(self, node):
        """Get neighbors of a node"""
        return list(self.graph.neighbors(node))
    
    def is_valid_node(self, node):
        """Check if node exists in graph"""
        return node in self.graph.nodes()


class MazeAdapter:
    """Adapter for Maze to work with dijkstra algorithm"""
    
    def __init__(self, maze: Maze):
        self.maze = maze
    
    def get_neighbors(self, node):
        """Get neighbors of a node in the maze"""
        return self.maze.get_neighbors(node[0], node[1])
    
    def is_valid_node(self, node):
        """Check if node is walkable in the maze"""
        return self.maze.is_walkable(node[0], node[1])


def find_random_start_goal(graph, max_attempts=100):
    """Find a valid start and goal pair in the graph"""
    nodes = list(graph.nodes())
    
    if len(nodes) < 2:
        return None, None
    
    for _ in range(max_attempts):
        start = random.choice(nodes)
        goal = random.choice(nodes)
        
        if start != goal and nx.has_path(graph, start, goal):
            return start, goal
    
    return None, None


def test_graph(name, graph, adapter):
    """Test Classic Dijkstra vs Bidirectional Dijkstra vs Bidirectional A* and compare"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    
    # Extract positions from the graph for A* heuristic
    positions = {}
    for node in graph.nodes():
        node_data = graph.nodes[node]
        if isinstance(node_data, dict):
            # For random_geometric_graph, extract x and y from the dict
            if 'x' in node_data and 'y' in node_data:
                positions[node] = (node_data['x'], node_data['y'])
            else:
                # Try to get first two numeric values
                values = [v for v in node_data.values() if isinstance(v, (int, float))]
                if len(values) >= 2:
                    positions[node] = tuple(values[:2])
                else:
                    positions[node] = (node, node)  # Fallback: use node index as coordinates
        else:
            # If it's already a tuple, use it directly
            if isinstance(node_data, (list, tuple)) and len(node_data) >= 2:
                positions[node] = tuple(node_data[:2])
            else:
                positions[node] = (node, node)  # Fallback
    
    # If positions dict is still empty or invalid, create default positions
    if not positions or all(isinstance(v, tuple) and v[0] == v[1] and isinstance(v[0], int) for v in positions.values()):
        # Use node indices as coordinates for fallback
        nodes_list = list(graph.nodes())
        positions = {node: (i, i) for i, node in enumerate(nodes_list)}
    
    # Find valid start and goal
    start, goal = find_random_start_goal(graph)
    
    if start is None:
        print("No valid start-goal pair found (disconnected component)")
        return None
    
    print(f"Start: {start}, Goal: {goal}\n")
    
    # ===== Test 1: Classic Dijkstra =====
    print("CLASSIC DIJKSTRA")
    start_time = time.time()
    path_classic, distance_classic, nodes_explored_classic = dijkstra_classique(adapter, start, goal)
    elapsed_classic = time.time() - start_time
    
    if path_classic is None:
        print("No path found")
        return None
    
    print(f"  Path length: {len(path_classic)} nodes | Distance: {distance_classic} | Nodes explored: {nodes_explored_classic} | Time: {elapsed_classic:.6f}s")
    
    # Visualize classic result
    print("Generating classic visualization...")
    filename_classic = f"figures/path_classic_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    visualize_path(graph, path_classic, start, goal, 
                   f"Classic Dijkstra\n{name}\nPath: {len(path_classic)} nodes | Distance: {distance_classic} | Nodes Explored: {nodes_explored_classic}", 
                   filename_classic)
    
    # ===== Test 2: Bidirectional Dijkstra =====
    print("\nBIDIRECTIONAL DIJKSTRA")
    start_time = time.time()
    path_bidir, distance_bidir, nodes_explored_bidir = dijkstra_bidirectionnel(adapter, start, goal)
    elapsed_bidir = time.time() - start_time
    
    if path_bidir is None:
        print("No path found")
        return None
    
    print(f"  Path length: {len(path_bidir)} nodes | Distance: {distance_bidir} | Nodes explored: {nodes_explored_bidir} | Time: {elapsed_bidir:.6f}s")
    
    # Visualize bidirectional result
    print("Generating bidirectional visualization...")
    filename_bidir = f"figures/path_bidirectional_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    visualize_path(graph, path_bidir, start, goal,
                   f"Bidirectional Dijkstra\n{name}\nPath: {len(path_bidir)} nodes | Distance: {distance_bidir} | Nodes Explored: {nodes_explored_bidir}",
                   filename_bidir)
    
    # ===== Test 3: Bidirectional A* =====
    print("\nBIDIRECTIONAL A*")
    start_time = time.time()
    path_astar, distance_astar, nodes_explored_astar = astar_bidirectionnel(adapter, start, goal, positions)
    elapsed_astar = time.time() - start_time
    
    if path_astar is None:
        print("No path found")
        return None
    
    print(f"  Path length: {len(path_astar)} nodes | Distance: {distance_astar} | Nodes explored: {nodes_explored_astar} | Time: {elapsed_astar:.6f}s")
    
    # Visualize A* result
    print("Generating A* visualization...")
    filename_astar = f"figures/path_astar_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    visualize_path(graph, path_astar, start, goal,
                   f"Bidirectional A*\n{name}\nPath: {len(path_astar)} nodes | Distance: {distance_astar} | Nodes Explored: {nodes_explored_astar}",
                   filename_astar)
    
    # ===== Test 4: NetworkX Bidirectional Dijkstra =====
    print("\nNETWORKX BIDIRECTIONAL DIJKSTRA")
    start_time = time.time()
    try:
        nx_distance, nx_path = nx.bidirectional_dijkstra(graph, start, goal)
        elapsed_nx = time.time() - start_time
        # Count nodes explored by computing distances from start
        forward_dist = nx.single_source_dijkstra_path_length(graph, start)
        nodes_explored_nx = sum(1 for node in forward_dist if forward_dist[node] <= nx_distance)
        print(f"  Path length: {len(nx_path)} nodes | Distance: {nx_distance} | Nodes explored: {nodes_explored_nx} | Time: {elapsed_nx:.6f}s")
        
        # Visualize NetworkX result
        print("Generating NetworkX visualization...")
        filename_nx = f"figures/path_networkx_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        visualize_path(graph, nx_path, start, goal,
                       f"NetworkX Bidirectional Dijkstra\n{name}\nPath: {len(nx_path)} nodes | Distance: {nx_distance} | Nodes Explored: {nodes_explored_nx}",
                       filename_nx)
    except nx.NetworkXNoPath:
        print("No path found")
        elapsed_nx = 0
        nodes_explored_nx = 0
        nx_distance = float('inf')
        nx_path = None
    
    return {
        'name': name,
        'type': 'graph',
        'nodes': graph.number_of_nodes(),
        'edges': graph.number_of_edges(),
        'nodes_explored_classic': nodes_explored_classic,
        'nodes_explored_bidir': nodes_explored_bidir,
        'nodes_explored_astar': nodes_explored_astar,
        'nodes_explored_nx': nodes_explored_nx,
        'time_classic': elapsed_classic,
        'time_bidir': elapsed_bidir,
        'time_astar': elapsed_astar,
        'time_nx': elapsed_nx,
        'distance': distance_classic
    }


def verify_path(graph, path):
    """Verify that path is valid in the graph"""
    for i in range(len(path) - 1):
        if not graph.has_edge(path[i], path[i + 1]):
            return False
    return True


def visualize_path(graph, path, start, goal, title, filename=None):
    """Visualize the graph and the found path"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use spring layout for better visualization
    try:
        pos = nx.spring_layout(graph, k=0.1, iterations=20, seed=42)
    except:
        pos = nx.spring_layout(graph, k=0.1, iterations=20)
    
    # Draw all edges in light gray
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='lightgray', width=0.5, alpha=0.5)
    
    # Draw path edges in red
    if path and len(path) > 1:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, ax=ax, 
                              edge_color='red', width=2.5, alpha=0.8)
    
    # Draw all nodes
    node_colors = []
    for node in graph.nodes():
        if node == start:
            node_colors.append('green')
        elif node == goal:
            node_colors.append('blue')
        elif path and node in path:
            node_colors.append('orange')
        else:
            node_colors.append('lightblue')
    
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, 
                          node_size=50, alpha=0.8)
    
    # Draw labels for start and goal only
    labels = {start: 'S', goal: 'G'}
    nx.draw_networkx_labels(graph, pos, labels, font_size=8, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    start_patch = mpatches.Patch(color='green', label='Start')
    goal_patch = mpatches.Patch(color='blue', label='Goal')
    path_patch = mpatches.Patch(color='orange', label='Path nodes')
    ax.legend(handles=[start_patch, goal_patch, path_patch], loc='upper left')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {filename}")
    
    plt.close()


def visualize_performance_comparison(results):
    """Visualize performance comparison across all tests"""
    
    # Separate maze and graph results
    maze_results = [r for r in results if r.get('type') == 'maze']
    graph_results = [r for r in results if r.get('type') != 'maze']
    
    if not results:
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- TIME COMPARISON ---
    ax1 = axes[0]
    test_names = [r['name'][:20] for r in results]
    classic_times = [r['time_classic'] * 1000 for r in results]  # Convert to milliseconds
    bidir_times = [r['time_bidir'] * 1000 for r in results]
    astar_times = [r['time_astar'] * 1000 for r in results]
    
    x = np.arange(len(test_names))
    width = 0.25
    
    bars1 = ax1.bar(x - width, classic_times, width, label='Classic Dijkstra', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x, bidir_times, width, label='Bidirectional Dijkstra', color='#4ECDC4', alpha=0.8)
    bars3 = ax1.bar(x + width, astar_times, width, label='Bidirectional A*', color='#45B7D1', alpha=0.8)
    
    ax1.set_xlabel('Test Cases', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8)
    
    # --- NODES EXPLORED COMPARISON ---
    ax2 = axes[1]
    classic_nodes = [r['nodes_explored_classic'] for r in results]
    bidir_nodes = [r['nodes_explored_bidir'] for r in results]
    astar_nodes = [r['nodes_explored_astar'] for r in results]
    
    bars1 = ax2.bar(x - width, classic_nodes, width, label='Classic Dijkstra', color='#FF6B6B', alpha=0.8)
    bars2 = ax2.bar(x, bidir_nodes, width, label='Bidirectional Dijkstra', color='#4ECDC4', alpha=0.8)
    bars3 = ax2.bar(x + width, astar_nodes, width, label='Bidirectional A*', color='#45B7D1', alpha=0.8)
    
    ax2.set_xlabel('Test Cases', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Nodes Explored', fontsize=12, fontweight='bold')
    ax2.set_title('Nodes Explored Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    filename = 'figures/performance_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Performance comparison graph saved to {filename}")
    plt.close()


def visualize_classic_vs_bidirectional(results):
    """Compare Classic Dijkstra vs Bidirectional Dijkstra"""
    graph_results = [r for r in results if r.get('type') != 'maze']
    
    if not graph_results:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create clear labels based on number of nodes
    test_names = [f"n={r['nodes']}" if r.get('type') != 'maze' else r['name'] for r in graph_results]
    classic_times = [r['time_classic'] * 1000 for r in graph_results]
    bidir_times = [r['time_bidir'] * 1000 for r in graph_results]
    classic_nodes = [r['nodes_explored_classic'] for r in graph_results]
    bidir_nodes = [r['nodes_explored_bidir'] for r in graph_results]
    
    x = np.arange(len(test_names))
    width = 0.35
    
    # Time comparison
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, classic_times, width, label='Classic Dijkstra', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, bidir_times, width, label='Bidirectional Dijkstra', color='#4ECDC4', alpha=0.8)
    
    ax1.set_xlabel('Test Cases', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Classic vs Bidirectional Dijkstra - Time', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Nodes explored comparison
    ax2 = axes[1]
    bars1 = ax2.bar(x - width/2, classic_nodes, width, label='Classic Dijkstra', color='#FF6B6B', alpha=0.8)
    bars2 = ax2.bar(x + width/2, bidir_nodes, width, label='Bidirectional Dijkstra', color='#4ECDC4', alpha=0.8)
    
    ax2.set_xlabel('Test Cases', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Nodes Explored', fontsize=12, fontweight='bold')
    ax2.set_title('Classic vs Bidirectional Dijkstra - Nodes Explored', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = 'figures/comparison_classic_vs_bidirectional.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Classic vs Bidirectional comparison saved to {filename}")
    plt.close()


def visualize_bidirectional_vs_astar(results):
    """Compare Bidirectional Dijkstra vs Bidirectional A*"""
    graph_results = [r for r in results if r.get('type') != 'maze']
    
    if not graph_results:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create clear labels based on number of nodes
    test_names = [f"n={r['nodes']}" if r.get('type') != 'maze' else r['name'] for r in graph_results]
    bidir_times = [r['time_bidir'] * 1000 for r in graph_results]
    astar_times = [r['time_astar'] * 1000 for r in graph_results]
    bidir_nodes = [r['nodes_explored_bidir'] for r in graph_results]
    astar_nodes = [r['nodes_explored_astar'] for r in graph_results]
    
    x = np.arange(len(test_names))
    width = 0.35
    
    # Time comparison
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, bidir_times, width, label='Bidirectional Dijkstra', color='#4ECDC4', alpha=0.8)
    bars2 = ax1.bar(x + width/2, astar_times, width, label='Bidirectional A*', color='#45B7D1', alpha=0.8)
    
    ax1.set_xlabel('Test Cases', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Bidirectional Dijkstra vs A* - Time', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Nodes explored comparison
    ax2 = axes[1]
    bars1 = ax2.bar(x - width/2, bidir_nodes, width, label='Bidirectional Dijkstra', color='#4ECDC4', alpha=0.8)
    bars2 = ax2.bar(x + width/2, astar_nodes, width, label='Bidirectional A*', color='#45B7D1', alpha=0.8)
    
    ax2.set_xlabel('Test Cases', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Nodes Explored', fontsize=12, fontweight='bold')
    ax2.set_title('Bidirectional Dijkstra vs A* - Nodes Explored', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = 'figures/comparison_bidirectional_vs_astar.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Bidirectional vs A* comparison saved to {filename}")
    plt.close()


def visualize_our_vs_networkx(results):
    """Compare our Bidirectional Dijkstra implementation vs NetworkX"""
    graph_results = [r for r in results if r.get('type') != 'maze' and 'nodes_explored_nx' in r]
    
    if not graph_results:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create clear labels based on number of nodes
    test_names = [f"n={r['nodes']}" if r.get('type') != 'maze' else r['name'] for r in graph_results]
    our_times = [r['time_bidir'] * 1000 for r in graph_results]
    nx_times = [r['time_nx'] * 1000 for r in graph_results]
    our_nodes = [r['nodes_explored_bidir'] for r in graph_results]
    nx_nodes = [r['nodes_explored_nx'] for r in graph_results]
    
    x = np.arange(len(test_names))
    width = 0.35
    
    # Time comparison
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, our_times, width, label='Our Implementation', color='#4ECDC4', alpha=0.8)
    bars2 = ax1.bar(x + width/2, nx_times, width, label='NetworkX', color='#95E1D3', alpha=0.8)
    
    ax1.set_xlabel('Test Cases', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Our Bidirectional Dijkstra vs NetworkX - Time', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Nodes explored comparison
    ax2 = axes[1]
    bars1 = ax2.bar(x - width/2, our_nodes, width, label='Our Implementation', color='#4ECDC4', alpha=0.8)
    bars2 = ax2.bar(x + width/2, nx_nodes, width, label='NetworkX', color='#95E1D3', alpha=0.8)
    
    ax2.set_xlabel('Test Cases', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Nodes Explored', fontsize=12, fontweight='bold')
    ax2.set_title('Our Bidirectional Dijkstra vs NetworkX - Nodes Explored', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = 'figures/comparison_our_vs_networkx.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Our vs NetworkX comparison saved to {filename}")
    plt.close()


def visualize_maze_path(maze, path, title, filename=None):
    """Visualize the maze and the found path"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw maze grid
    for i in range(maze.height):
        for j in range(maze.width):
            if maze.grid[i][j] == 1:  # Obstacle
                rect = mpatches.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=1, 
                                         edgecolor='black', facecolor='black')
                ax.add_patch(rect)
            else:  # Walkable
                rect = mpatches.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=0.5, 
                                         edgecolor='gray', facecolor='white')
                ax.add_patch(rect)
    
    # Draw path
    if path and len(path) > 1:
        for node in path[1:-1]:  # Exclude start and goal
            circle = mpatches.Circle((node[1], node[0]), 0.2, color='orange', alpha=0.8)
            ax.add_patch(circle)
    
    # Draw start and goal
    start = maze.start
    goal = maze.goal
    start_circle = mpatches.Circle((start[1], start[0]), 0.3, color='green', alpha=0.9)
    goal_circle = mpatches.Circle((goal[1], goal[0]), 0.3, color='blue', alpha=0.9)
    ax.add_patch(start_circle)
    ax.add_patch(goal_circle)
    
    # Draw path line
    if path and len(path) > 1:
        path_y = [node[1] for node in path]
        path_x = [node[0] for node in path]
        ax.plot(path_y, path_x, 'r-', linewidth=2, alpha=0.6)
    
    ax.set_xlim(-1, maze.width)
    ax.set_ylim(maze.height, -1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Add legend
    start_patch = mpatches.Patch(color='green', label='Start')
    goal_patch = mpatches.Patch(color='blue', label='Goal')
    path_patch = mpatches.Patch(color='orange', label='Path')
    obstacle_patch = mpatches.Patch(color='black', label='Obstacle')
    ax.legend(handles=[start_patch, goal_patch, path_patch, obstacle_patch], loc='upper right')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {filename}")
    
    plt.close()


def test_maze(maze, adapter, name="Simple Maze"):
    """Test Classic Dijkstra vs Bidirectional Dijkstra vs Bidirectional A* on maze"""
    print(f"\n{'='*60}")
    print(f"Testing Maze: {name}")
    print(f"{'='*60}")
    print(f"Maze size: {maze.height}x{maze.width}")
    print(f"Start: {maze.start}, Goal: {maze.goal}\n")
    
    start = maze.start
    goal = maze.goal
    
    # Create positions dict for A* heuristic (grid positions)
    positions = {(i, j): (i, j) for i in range(maze.height) for j in range(maze.width)}
    
    # ===== Test 1: Classic Dijkstra =====
    print("CLASSIC DIJKSTRA")
    start_time = time.time()
    path_classic, distance_classic, nodes_explored_classic = dijkstra_classique(adapter, start, goal)
    elapsed_classic = time.time() - start_time
    
    if path_classic is None:
        print("No path found")
        return None
    
    print(f"  Path length: {len(path_classic)} nodes | Distance: {distance_classic} | Nodes explored: {nodes_explored_classic} | Time: {elapsed_classic:.6f}s")
    
    # Visualize classic result
    print("Generating classic visualization...")
    filename_classic = f"figures/maze_classic_{name.replace(' ', '_')}.png"
    visualize_maze_path(maze, path_classic, f"Classic Dijkstra\n{name}\nPath: {len(path_classic)} nodes", filename_classic)
    
    # ===== Test 2: Bidirectional Dijkstra =====
    print("\nBIDIRECTIONAL DIJKSTRA")
    start_time = time.time()
    path_bidir, distance_bidir, nodes_explored_bidir = dijkstra_bidirectionnel(adapter, start, goal)
    elapsed_bidir = time.time() - start_time
    
    if path_bidir is None:
        print("No path found")
        return None
    
    print(f"  Path length: {len(path_bidir)} nodes | Distance: {distance_bidir} | Nodes explored: {nodes_explored_bidir} | Time: {elapsed_bidir:.6f}s")
    
    # Visualize bidirectional result
    print("Generating bidirectional visualization...")
    filename_bidir = f"figures/maze_bidirectional_{name.replace(' ', '_')}.png"
    visualize_maze_path(maze, path_bidir, f"Bidirectional Dijkstra\n{name}\nPath: {len(path_bidir)} nodes", filename_bidir)
    
    # ===== Test 3: Bidirectional A* =====
    print("\nBIDIRECTIONAL A*")
    start_time = time.time()
    path_astar, distance_astar, nodes_explored_astar = astar_bidirectionnel(adapter, start, goal, positions)
    elapsed_astar = time.time() - start_time
    
    if path_astar is None:
        print("No path found")
        return None
    
    print(f"  Path length: {len(path_astar)} nodes | Distance: {distance_astar} | Nodes explored: {nodes_explored_astar} | Time: {elapsed_astar:.6f}s")
    
    # Visualize A* result
    print("Generating A* visualization...")
    filename_astar = f"figures/maze_astar_{name.replace(' ', '_')}.png"
    visualize_maze_path(maze, path_astar, f"Bidirectional A*\n{name}\nPath: {len(path_astar)} nodes", filename_astar)
    
    return {
        'name': name,
        'type': 'maze',
        'size': f"{maze.height}x{maze.width}",
        'nodes_explored_classic': nodes_explored_classic,
        'nodes_explored_bidir': nodes_explored_bidir,
        'nodes_explored_astar': nodes_explored_astar,
        'time_classic': elapsed_classic,
        'time_bidir': elapsed_bidir,
        'time_astar': elapsed_astar,
        'distance': distance_classic
    }


def main():
    print("\n" + "="*60)
    print("DIJKSTRA & A* COMPARISON - TESTS ON VARIOUS GRAPHS")
    print("="*60)
    
    results = []
    
    # Test 0: Small Grid Graph (10x10)
    print("\n" + "="*60)
    print("TEST 0: Small Grid Graph (10x10)")
    print("="*60)
    print("\nGenerating Small Grid Graph...")
    G_small = nx.grid_2d_graph(10, 10)
    # Convert to simple integer node labels
    G_small = nx.convert_node_labels_to_integers(G_small)
    print(f"Generated: {G_small.number_of_nodes()} nodes, {G_small.number_of_edges()} edges")
    adapter_small = GraphAdapter(G_small)
    result_small = test_graph("Grid Graph (10x10)", G_small, adapter_small)
    if result_small:
        results.append(result_small)
    
    # Test: Large Random Geometric Graph (n=2000, radius=0.05)
    print("\n" + "="*60)
    print("TEST: Large Random Geometric Graph")
    print("="*60)
    print("\nGenerating Large Random Geometric Graph...")
    G_large = nx.random_geometric_graph(2000, 0.05)
    print(f"Generated: {G_large.number_of_nodes()} nodes, {G_large.number_of_edges()} edges")
    adapter_large = GraphAdapter(G_large)
    result_large = test_graph("Random Geometric Graph (n=2000, radius=0.05)", G_large, adapter_large)
    if result_large:
        results.append(result_large)
    
    # Summary and Comparison
    if results:
        # Filter to show tests for small and large graphs
        filtered_results = [r for r in results if r.get('type') == 'graph']
        
        print("\n" + "="*80)
        print("SUMMARY - DIJKSTRA vs BIDIRECTIONAL DIJKSTRA vs BIDIRECTIONAL A* COMPARISON")
        print("="*80)
        print(f"{'Graph':<35} {'Classic':<15} {'Bidir':<15} {'A* Bidir':<15}")
        print(f"{'':35} {'Nodes Exp':<15} {'Nodes Exp':<15} {'Nodes Exp':<15}")
        print("-"*80)
        for result in filtered_results:
            graph_name = result['name'][:32]
            classic_nodes = result['nodes_explored_classic']
            bidir_nodes = result['nodes_explored_bidir']
            astar_nodes = result['nodes_explored_astar']
            print(f"{graph_name:<35} {classic_nodes:<15} {bidir_nodes:<15} {astar_nodes:<15}")
        
        print("\n" + "="*80)
        print("PERFORMANCE METRICS (TIME)")
        print("="*80)
        print(f"{'Graph':<35} {'Classic':<15} {'Bidir':<15} {'A* Bidir':<15}")
        print(f"{'':35} {'(seconds)':<15} {'(seconds)':<15} {'(seconds)':<15}")
        print("-"*80)
        for result in filtered_results:
            graph_name = result['name'][:32]
            classic_time = result['time_classic']
            bidir_time = result['time_bidir']
            astar_time = result['time_astar']
            print(f"{graph_name:<35} {classic_time:<15.6f} {bidir_time:<15.6f} {astar_time:<15.6f}")
        
        print("\n" + "="*80)
        print("EFFICIENCY ANALYSIS")
        print("="*80)
        for result in filtered_results:
            improvement_bidir = ((result['nodes_explored_classic'] - result['nodes_explored_bidir']) / result['nodes_explored_classic']) * 100
            improvement_astar = ((result['nodes_explored_classic'] - result['nodes_explored_astar']) / result['nodes_explored_classic']) * 100
            speedup_bidir = result['time_classic'] / result['time_bidir'] if result['time_bidir'] > 0 else float('inf')
            speedup_astar = result['time_classic'] / result['time_astar'] if result['time_astar'] > 0 else float('inf')
            graph_name = result['name']
            print(f"\n{graph_name}:")
            print(f"  Bidirectional Dijkstra:")
            print(f"    Nodes explored reduction: {improvement_bidir:.2f}%")
            print(f"    Speedup factor: {speedup_bidir:.2f}x")
            print(f"  Bidirectional A*:")
            print(f"    Nodes explored reduction: {improvement_astar:.2f}%")
            print(f"    Speedup factor: {speedup_astar:.2f}x")
        
        print("\n" + "="*80)
        print("GENERATED VISUALIZATIONS")
        print("="*80)
        print("All visualizations have been saved to the 'figures/' directory:")
        for result in results:
            graph_name = result['name'].replace(' ', '_').replace('(', '').replace(')', '')
            classic_file = f"figures/path_classic_{graph_name}.png"
            bidir_file = f"figures/path_bidirectional_{graph_name}.png"
            astar_file = f"figures/path_astar_{graph_name}.png"
            networkx_file = f"figures/path_networkx_{graph_name}.png"
            print(f"\n{result['name']} (n={result['nodes']}, edges={result['edges']}):")
            print(f"  ✓ {classic_file}")
            print(f"  ✓ {bidir_file}")
            print(f"  ✓ {astar_file}")
            print(f"  ✓ {networkx_file}")
        
        # Create performance comparison graphs
        print("\n" + "="*80)
        print("GENERATING COMPARISON FIGURES")
        print("="*80)
        visualize_performance_comparison(results)
        visualize_classic_vs_bidirectional(results)
        visualize_bidirectional_vs_astar(results)
        visualize_our_vs_networkx(results)
        print("\n✓ All comparison figures have been generated and saved to the 'figures/' directory")


if __name__ == "__main__":
    main()