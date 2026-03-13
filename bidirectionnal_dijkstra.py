import networkx as nx
import time

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def visualize_shortest_path(graph, path, pos=None, title="Shortest path", save_path=None):
    """Visualise un graphe NetworkX et le chemin trouvé.

    :param graph: le graphe NetworkX.
    :param path: liste des nœuds formant le chemin (ordre start->goal).
    :param pos: positions des nœuds (dict node -> (x,y)). Si None, utilise une mise en page.
    :param title: titre de la fenêtre.
    :param save_path: si fourni, enregistre l'image au lieu de l'afficher.
    """
    if not HAS_MATPLOTLIB:
        print("[ERREUR] matplotlib n'est pas installé. Installez avec: pip install matplotlib")
        return

    if pos is None:
        # Si le graphe a des positions (ex. random_geometric_graph), les utiliser.
        pos = nx.get_node_attributes(graph, "pos")
        if not pos:
            pos = nx.spring_layout(graph)

    # Tracer tout le graphe en gris clair
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(graph, pos, node_size=30, node_color="lightgray", alpha=0.7)
    nx.draw_networkx_edges(graph, pos, edge_color="lightgray", alpha=0.5)

    # Tracer le chemin
    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        # Arêtes du chemin en rouge
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color="red", width=2)

        # Nœuds du chemin : début (vert), fin (bleu), intermédiaires (rouge)
        start_node, end_node = path[0], path[-1]
        mid_nodes = path[1:-1] if len(path) > 2 else []

        nx.draw_networkx_nodes(graph, pos, nodelist=mid_nodes, node_size=80, node_color="red")
        nx.draw_networkx_nodes(graph, pos, nodelist=[start_node], node_size=120, node_color="green")
        nx.draw_networkx_nodes(graph, pos, nodelist=[end_node], node_size=120, node_color="blue")

        # Légende simple
        plt.scatter([], [], c="green", label="Début", s=100)
        plt.scatter([], [], c="blue", label="Fin", s=100)
        plt.scatter([], [], c="red", label="Chemin", s=100)
        plt.legend(loc="best")

    plt.title(title)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Graphique enregistré dans: {save_path}")
        plt.close()
    else:
        plt.show()


def dijkstra_bidirectional(graph, start, goal):
    # Initialize the forward and backward search
    forward_queue = [(0, start)]
    backward_queue = [(0, goal)]
    forward_distances = {start: 0}
    backward_distances = {goal: 0}
    forward_predecessors = {start: None}
    backward_predecessors = {goal: None}

    while forward_queue and backward_queue:
        # Expand the forward search
        if forward_queue:
            current_forward_distance, current_forward_node = forward_queue.pop(0)
            for neighbor in graph.neighbors(current_forward_node):
                distance = current_forward_distance + graph[current_forward_node][neighbor].get('weight', 1)
                if neighbor not in forward_distances or distance < forward_distances[neighbor]:
                    forward_distances[neighbor] = distance
                    forward_predecessors[neighbor] = current_forward_node
                    forward_queue.append((distance, neighbor))

        # Expand the backward search
        if backward_queue:
            current_backward_distance, current_backward_node = backward_queue.pop(0)
            for neighbor in graph.neighbors(current_backward_node):
                distance = current_backward_distance + graph[current_backward_node][neighbor].get('weight', 1)
                if neighbor not in backward_distances or distance < backward_distances[neighbor]:
                    backward_distances[neighbor] = distance
                    backward_predecessors[neighbor] = current_backward_node
                    backward_queue.append((distance, neighbor))

        # Check for intersection of the two searches
        intersection_nodes = set(forward_distances.keys()) & set(backward_distances.keys())
        if intersection_nodes:
            # Find the node with the minimum total distance from both searches
            min_distance = float('inf')
            meeting_node = None
            for node in intersection_nodes:
                total_distance = forward_distances[node] + backward_distances[node]
                if total_distance < min_distance:
                    min_distance = total_distance
                    meeting_node = node

            # Reconstruct the path from start to meeting node and from meeting node to goal
            path_forward = []
            node = meeting_node
            while node is not None:
                path_forward.append(node)
                node = forward_predecessors[node]
            path_forward.reverse()

            path_backward = []
            node = meeting_node
            while node is not None:
                path_backward.append(node)
                node = backward_predecessors[node]

            return path_forward + path_backward[1:]
        

def dijkstra_with_stats(graph, start, goal):
    """Dijkstra classique (unidirectionnel) avec statistiques.

    Retourne un tuple (path, nodes_explored, duration_s).
    """
    import heapq

    t0 = time.time()
    open_heap = [(0, start)]
    dist = {start: 0}
    prev = {start: None}
    explored = 0

    while open_heap:
        d, u = heapq.heappop(open_heap)
        if d > dist.get(u, float('inf')):
            continue
        explored += 1
        if u == goal:
            break
        for v in graph.neighbors(u):
            alt = d + graph[u][v].get('weight', 1)
            if alt < dist.get(v, float('inf')):
                dist[v] = alt
                prev[v] = u
                heapq.heappush(open_heap, (alt, v))

    # Reconstruction du chemin
    path = []
    node = goal
    while node is not None and node in prev:
        path.append(node)
        node = prev[node]
    path.reverse()

    return path, explored, time.time() - t0


def bidirectional_dijkstra_with_stats(graph, start, goal):
    """Bidirectional Dijkstra (algorithme maison) avec statistique d'exploration."""
    forward_queue = [(0, start)]
    backward_queue = [(0, goal)]
    forward_distances = {start: 0}
    backward_distances = {goal: 0}
    forward_prev = {start: None}
    backward_prev = {goal: None}
    explored_forward = 0
    explored_backward = 0
    t0 = time.time()

    while forward_queue and backward_queue:
        # Forward
        if forward_queue:
            d, u = forward_queue.pop(0)
            if d > forward_distances.get(u, float('inf')):
                continue
            explored_forward += 1
            for v in graph.neighbors(u):
                alt = d + graph[u][v].get('weight', 1)
                if alt < forward_distances.get(v, float('inf')):
                    forward_distances[v] = alt
                    forward_prev[v] = u
                    forward_queue.append((alt, v))

        # Backward
        if backward_queue:
            d, u = backward_queue.pop(0)
            if d > backward_distances.get(u, float('inf')):
                continue
            explored_backward += 1
            for v in graph.neighbors(u):
                alt = d + graph[u][v].get('weight', 1)
                if alt < backward_distances.get(v, float('inf')):
                    backward_distances[v] = alt
                    backward_prev[v] = u
                    backward_queue.append((alt, v))

        # Intersection
        intersect = set(forward_distances) & set(backward_distances)
        if intersect:
            best = min(intersect, key=lambda n: forward_distances[n] + backward_distances[n])

            # Reconstruction
            path_f = []
            node = best
            while node is not None:
                path_f.append(node)
                node = forward_prev.get(node)
            path_f.reverse()

            path_b = []
            node = best
            while node is not None:
                path_b.append(node)
                node = backward_prev.get(node)

            path = path_f + path_b[1:]
            return path, explored_forward + explored_backward, time.time() - t0

    return None, explored_forward + explored_backward, time.time() - t0


def bidirectional_astar_with_stats(graph, start, goal, pos=None):
    """A* bidirectionnel (heuristique euclidienne) avec statistique d'exploration."""
    import heapq

    if pos is None:
        pos = nx.get_node_attributes(graph, "pos")
        if not pos:
            pos = nx.spring_layout(graph)

    def h(u, v):
        ux, uy = pos[u]
        vx, vy = pos[v]
        return ((ux - vx) ** 2 + (uy - vy) ** 2) ** 0.5

    t0 = time.time()
    g_f = {start: 0.0}
    g_b = {goal: 0.0}
    prev_f = {start: None}
    prev_b = {goal: None}
    open_f = [(h(start, goal), start)]
    open_b = [(h(goal, start), goal)]
    closed_f = set()
    closed_b = set()
    explored = 0

    best_meeting = None

    while open_f and open_b:
        # Expander côté forward
        if open_f:
            _, u = heapq.heappop(open_f)
            if u in closed_f:
                continue
            closed_f.add(u)
            explored += 1
            if u in closed_b:
                best_meeting = u
                break
            for v in graph.neighbors(u):
                tentative = g_f[u] + graph[u][v].get('weight', 1)
                if tentative < g_f.get(v, float('inf')):
                    g_f[v] = tentative
                    prev_f[v] = u
                    heapq.heappush(open_f, (tentative + h(v, goal), v))

        # Expander côté backward
        if open_b:
            _, u = heapq.heappop(open_b)
            if u in closed_b:
                continue
            closed_b.add(u)
            explored += 1
            if u in closed_f:
                best_meeting = u
                break
            for v in graph.neighbors(u):
                tentative = g_b[u] + graph[u][v].get('weight', 1)
                if tentative < g_b.get(v, float('inf')):
                    g_b[v] = tentative
                    prev_b[v] = u
                    heapq.heappush(open_b, (tentative + h(v, start), v))

    if best_meeting is None:
        return None, explored, time.time() - t0

    # Reconstruction du chemin
    path_f = []
    node = best_meeting
    while node is not None:
        path_f.append(node)
        node = prev_f.get(node)
    path_f.reverse()

    path_b = []
    node = best_meeting
    while node is not None:
        path_b.append(node)
        node = prev_b.get(node)

    path = path_f + path_b[1:]
    return path, explored, time.time() - t0


def nx_bidirectional_dijkstra(graph, start, goal):
    """Appel de la fonction intégrée de NetworkX pour bidirectional_dijkstra."""
    from networkx.algorithms.shortest_paths.weighted import bidirectional_dijkstra

    t0 = time.time()
    dist, path = bidirectional_dijkstra(graph, start, goal)
    return path, dist, time.time() - t0


def compare_algorithms():
    """Fait tourner plusieurs algorithmes et compare leurs métriques."""
    num_nodes = 500
    graph = nx.random_geometric_graph(num_nodes, 0.15)
    start = 0
    goal = num_nodes - 1

    print("\n=== Comparaison : NetworkX bidirectional_dijkstra vs notre implémentation ===")
    path_nx, dist_nx, t_nx = nx_bidirectional_dijkstra(graph, start, goal)
    print(f"NetworkX : longueur={len(path_nx)-1}, coût={dist_nx:.2f}, temps={t_nx:.3f}s")

    path_bi, explored_bi, t_bi = bidirectional_dijkstra_with_stats(graph, start, goal)
    print(f"Bidirectional Dijkstra (ma version) : longueur={len(path_bi)-1}, noeuds explorés={explored_bi}, temps={t_bi:.3f}s")

    print("\n=== Comparaison : Dijkstra unidirectionnel vs Dijkstra bidirectionnel ===")
    path_uni, explored_uni, t_uni = dijkstra_with_stats(graph, start, goal)
    print(f"Dijkstra unidirectionnel : longueur={len(path_uni)-1}, noeuds explorés={explored_uni}, temps={t_uni:.3f}s")
    print(f"Dijkstra bidirectionnel : longueur={len(path_bi)-1}, noeuds explorés={explored_bi}, temps={t_bi:.3f}s")

    print("\n=== Comparaison : Dijkstra bidirectionnel vs A* bidirectionnel ===")
    path_astar, explored_astar, t_astar = bidirectional_astar_with_stats(graph, start, goal)
    print(f"A* bidirectionnel : longueur={len(path_astar)-1 if path_astar else 'N/A'}, noeuds explorés={explored_astar}, temps={t_astar:.3f}s")

    # Visualisation du chemin trouvé par A* bidirectionnel
    visualize_shortest_path(graph, path_astar or path_bi, title="A* bidirectionnel (ou Dijkstra si A* échoue)")


if __name__ == "__main__":
    compare_algorithms()



