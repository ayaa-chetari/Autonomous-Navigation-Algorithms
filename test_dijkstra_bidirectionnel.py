import networkx as nx
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dijkstra_bidirectionnel import dijkstra_bidirectionnel


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
    """Test Dijkstra bidirectional on a specific graph"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    
    # Find valid start and goal
    start, goal = find_random_start_goal(graph)
    
    if start is None:
        print("No valid start-goal pair found (disconnected component)")
        return None
    
    print(f"Start: {start}, Goal: {goal}")
    
    # Run algorithm
    start_time = time.time()
    path, distance = dijkstra_bidirectionnel(adapter, start, goal)
    elapsed = time.time() - start_time
    
    if path is None:
        print("No path found")
        return None
    
    print(f"Path length: {len(path)} nodes")
    print(f"Distance: {distance}")
    print(f"Execution time: {elapsed:.4f} seconds")
    
    # Verify path
    if verify_path(graph, path):
        print("✓ Path verified")
    else:
        print("✗ Path verification failed")
    
    # Visualize
    print("Generating visualization...")
    filename = f"path_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    visualize_path(graph, path, start, goal, f"{name}\nPath Length: {len(path)}, Distance: {distance}", filename)
    
    return {
        'name': name,
        'nodes': graph.number_of_nodes(),
        'edges': graph.number_of_edges(),
        'path_length': len(path),
        'distance': distance,
        'time': elapsed
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
    
    plt.show()
    plt.close()


def main():
    print("\n" + "="*60)
    print("BIDIRECTIONAL DIJKSTRA - GRAPH TESTS")
    print("="*60)
    
    results = []
    
    # Test 1: Random Geometric Graph
    print("\nGenerating Random Geometric Graph...")
    G_geometric = nx.random_geometric_graph(2000, 0.05)
    print(f"Generated: {G_geometric.number_of_nodes()} nodes, {G_geometric.number_of_edges()} edges")
    adapter_geometric = GraphAdapter(G_geometric)
    result1 = test_graph("Random Geometric Graph (n=2000, radius=0.05)", G_geometric, adapter_geometric)
    if result1:
        results.append(result1)
    
    # Test 2: Erdős–Rényi Graph
    print("\nGenerating Erdős–Rényi Graph...")
    G_erdos_renyi = nx.erdos_renyi_graph(1000, 0.01)
    print(f"Generated: {G_erdos_renyi.number_of_nodes()} nodes, {G_erdos_renyi.number_of_edges()} edges")
    adapter_erdos_renyi = GraphAdapter(G_erdos_renyi)
    result2 = test_graph("Erdős–Rényi Graph (n=1000, p=0.01)", G_erdos_renyi, adapter_erdos_renyi)
    if result2:
        results.append(result2)
    
    # Test 3: Barabási–Albert Graph (Scale-free)
    print("\nGenerating Barabási–Albert Graph...")
    G_barabasi_albert = nx.barabasi_albert_graph(1000, 5)
    print(f"Generated: {G_barabasi_albert.number_of_nodes()} nodes, {G_barabasi_albert.number_of_edges()} edges")
    adapter_barabasi_albert = GraphAdapter(G_barabasi_albert)
    result3 = test_graph("Barabási–Albert Graph (n=1000, m=5)", G_barabasi_albert, adapter_barabasi_albert)
    if result3:
        results.append(result3)
    
    # Summary
    if results:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"{'Graph':<40} {'Time (s)':<12} {'Path Length':<12}")
        print("-"*60)
        for result in results:
            print(f"{result['name']:<40} {result['time']:<12.4f} {result['path_length']:<12}")


if __name__ == "__main__":
    main()
