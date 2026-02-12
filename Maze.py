import heapq
import math
from typing import List, Tuple, Optional, Dict
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class Maze:
    def __init__(self, height: int, width: int, start: Tuple[int, int], goal: Tuple[int, int]):
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal

        self.grid: List[List[int]] = [[0 for _ in range(width)] for _ in range(height)]
        self.reward: List[List[float]] = [[0.0 for _ in range(width)] for _ in range(height)]

        if not self.in_bounds(*start) or not self.in_bounds(*goal):
            raise ValueError("Start ou Goal hors de la grille")

        self.grid[start[0]][start[1]] = 0
        self.grid[goal[0]][goal[1]] = 0

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.height and 0 <= y < self.width

    def is_walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.grid[x][y] == 0

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """4 directions (haut, bas, gauche, droite)."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors: List[Tuple[int, int]] = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_walkable(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def get_neighbors_with_diagonals(self, x: int, y: int) -> List[Tuple[int, int]]:
        """8 directions : 4 orthogonales + 4 diagonales."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # orthogonal
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]  # diagonal
        neighbors: List[Tuple[int, int]] = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_walkable(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def add_obstacle(self, x: int, y: int) -> None:
        if (x, y) != self.start and (x, y) != self.goal and self.in_bounds(x, y):
            self.grid[x][y] = 1

    def clear_obstacles(self) -> None:
        for i in range(self.height):
            for j in range(self.width):
                self.grid[i][j] = 0

    # Obstacles fixes 

    def set_fixed_obstacles(self) -> None:
        self.clear_obstacles()

        for i in range(self.height):
            if i not in (1, 6):
                self.add_obstacle(i, 3)

        for i in range(self.height):
            if i not in (2, 5):
                self.add_obstacle(i, 7)

        for j in range(8, 11):
            if j not in (9, 10):
                self.add_obstacle(4, j)

        extra_blocks = [(1, 5), (2, 5), (5, 9), (6, 10), (3, 9)]
        for x, y in extra_blocks:
            self.add_obstacle(x, y)

        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]] = 0

  
    # A* Algorithm implementation

    def heuristic_manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def solve(self) -> Optional[List[Tuple[int, int]]]:
        """A* avec 4 directions (orthogonales)."""
        start = self.start
        goal = self.goal

        open_heap = []
        heapq.heappush(open_heap, (self.heuristic_manhattan(start, goal), 0, start))

        g_score = {start: 0}
        came_from = {}
        closed_set = set()

        while open_heap:
            _, g_current, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(*current):
                if neighbor in closed_set:
                    continue

                tentative_g = g_current + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic_manhattan(neighbor, goal)
                    heapq.heappush(open_heap, (f, tentative_g, neighbor))

        return None

    def dijkstra(self) -> Optional[List[Tuple[int, int]]]:
     
        start = self.start
        goal = self.goal

        # Heap : (g, (x,y))  sans f (heuristique = 0)
        open_heap = []
        heapq.heappush(open_heap, (0, start))

        g_score = {start: 0}
        came_from = {}
        closed_set = set()

        while open_heap:
            g_current, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(*current):
                if neighbor in closed_set:
                    continue

                tentative_g = g_current + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_heap, (tentative_g, neighbor))

        return None

    def solve_with_stats(self) -> tuple:
        """A* avec statistiques (chemin, nœuds exploré, temps)."""
        import time
        start_time = time.time()
        start = self.start
        goal = self.goal

        open_heap = []
        heapq.heappush(open_heap, (self.heuristic_manhattan(start, goal), 0, start))

        g_score = {start: 0}
        came_from = {}
        closed_set = set()
        nodes_explored = 0

        while open_heap:
            _, g_current, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue
            closed_set.add(current)
            nodes_explored += 1

            if current == goal:
                elapsed = time.time() - start_time
                return self.reconstruct_path(came_from, current), nodes_explored, elapsed

            for neighbor in self.get_neighbors(*current):
                if neighbor in closed_set:
                    continue

                tentative_g = g_current + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic_manhattan(neighbor, goal)
                    heapq.heappush(open_heap, (f, tentative_g, neighbor))

        return None, nodes_explored, time.time() - start_time

    def dijkstra_with_stats(self) -> tuple:
        """Dijkstra avec statistiques (chemin, nœuds exploré, temps)."""
        import time
        start_time = time.time()
        start = self.start
        goal = self.goal

        open_heap = []
        heapq.heappush(open_heap, (0, start))

        g_score = {start: 0}
        came_from = {}
        closed_set = set()
        nodes_explored = 0

        while open_heap:
            g_current, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue
            closed_set.add(current)
            nodes_explored += 1

            if current == goal:
                elapsed = time.time() - start_time
                return self.reconstruct_path(came_from, current), nodes_explored, elapsed

            for neighbor in self.get_neighbors(*current):
                if neighbor in closed_set:
                    continue

                tentative_g = g_current + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_heap, (tentative_g, neighbor))

        return None, nodes_explored, time.time() - start_time

    def compare_algorithms(self) -> None:
        """Compare A* et Dijkstra en termes de performances."""
        print("\n" + "="*60)
        print("COMPARAISON: A* vs Dijkstra")
        print("="*60)
        
        path_astar, nodes_astar, time_astar = self.solve_with_stats()
        path_dijkstra, nodes_dijkstra, time_dijkstra = self.dijkstra_with_stats()
        
        print(f"\nA*:")
        print(f"  Longueur du chemin : {len(path_astar) - 1 if path_astar else 'N/A'}")
        print(f"  Noeuds explorés : {nodes_astar}")
        
        print(f"\nDijkstra:")
        print(f"  Longueur du chemin : {len(path_dijkstra) - 1 if path_dijkstra else 'N/A'}")
        print(f"  Noeuds explorés : {nodes_dijkstra}")
        
        print(f"\nDifférence:")
        print(f"  Réduction nœuds explorés : {(1 - nodes_astar/nodes_dijkstra)*100:.1f}%")
        print("="*60)

    def heuristic_euclidean(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Distance euclidienne adaptée aux diagonales."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return dx + dy

    def movement_cost(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Coût du mouvement entre deux cellules (1.0 orthogonal, sqrt(2) diagonal)."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx + dy == 1:  # orthogonal
            return 1.0
        elif dx == 1 and dy == 1:  # diagonal
            return math.sqrt(2)
        return 1.0

    def solve_with_diagonals(self) -> Optional[List[Tuple[int, int]]]:
        """A* avec 8 directions (orthogonales + diagonales) et coûts adaptés."""
        start = self.start
        goal = self.goal

        open_heap = []
        h_start = self.heuristic_euclidean(start, goal)
        heapq.heappush(open_heap, (h_start, 0.0, start))

        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set: set = set()

        while open_heap:
            _, g_current, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            cx, cy = current
            for neighbor in self.get_neighbors_with_diagonals(cx, cy):
                if neighbor in closed_set:
                    continue

                nx, ny = neighbor
                cost = self.movement_cost(cx, cy, nx, ny)
                tentative_g = g_current + cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = self.heuristic_euclidean(neighbor, goal)
                    f = tentative_g + h
                    heapq.heappush(open_heap, (f, tentative_g, neighbor))

        return None

    def init_rewards(self, step_penalty: float = -1.0, goal_reward: float = 100.0,
                     bonus_cells: Optional[Dict[Tuple[int, int], float]] = None) -> None:
        """Initialise la matrice des récompenses.

        Par défaut applique une pénalité uniforme à chaque déplacement, ajoute des bonus si fournis,
        et met une récompense significative sur la cellule d'arrivée.
        """
        for i in range(self.height):
            for j in range(self.width):
                self.reward[i][j] = 0.0 if self.grid[i][j] == 1 else step_penalty

        if bonus_cells:
            for (x, y), val in bonus_cells.items():
                if self.in_bounds(x, y):
                    self.reward[x][y] = val

        gx, gy = self.goal
        self.reward[gx][gy] = goal_reward

    def display_with_path(self, path):
        """Affiche le labyrinthe avec le chemin."""
        path_set = set(path) if path else set()
        for i in range(self.height):
            row = ""
            for j in range(self.width):
                cell = (i, j)
                if cell == self.start:
                    row += "S "
                elif cell == self.goal:
                    row += "G "
                elif self.grid[i][j] == 1:
                    row += "# "
                elif cell in path_set:
                    row += "* "
                else:
                    row += ". "
            print(row)

    def display(self):
        """Affiche le labyrinthe simple (sans chemin)."""
        for i in range(self.height):
            row = ""
            for j in range(self.width):
                cell = (i, j)
                if cell == self.start:
                    row += "S "
                elif cell == self.goal:
                    row += "G "
                elif self.grid[i][j] == 1:
                    row += "# "
                else:
                    row += ". "
            print(row)

    def visualize(self, path: Optional[List[Tuple[int, int]]] = None, title: str = "Maze") -> None:
        """Visualise le labyrinthe et le chemin en graphique (matplotlib).

        :param path: chemin à afficher (liste de tuples (x, y))
        :param title: titre du graphique
        """
        if not HAS_MATPLOTLIB:
            print("[ERREUR] matplotlib n'est pas installé.")
            print("  Installez avec: pip install matplotlib")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Afficher les cellules
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i][j] == 1:
                    # Obstacle (noir)
                    rect = patches.Rectangle((j, self.height - 1 - i), 1, 1,
                                           linewidth=0.5, edgecolor='gray', facecolor='black')
                    ax.add_patch(rect)
                else:
                    # Cellule libre (blanc avec grille)
                    rect = patches.Rectangle((j, self.height - 1 - i), 1, 1,
                                           linewidth=0.5, edgecolor='lightgray', facecolor='white')
                    ax.add_patch(rect)

        # Afficher le chemin
        if path:
            path_x = [p[1] + 0.5 for p in path]
            path_y = [self.height - 0.5 - p[0] for p in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path', alpha=0.7)
            ax.plot(path_x, path_y, 'bo', markersize=4)

        # Afficher start et goal
        sx, sy = self.start
        gx, gy = self.goal
        ax.plot(sy + 0.5, self.height - 0.5 - sx, 'go', markersize=12, label='Start', zorder=5)
        ax.plot(gy + 0.5, self.height - 0.5 - gx, 'r*', markersize=20, label='Goal', zorder=5)

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_xlabel('Column (Y)')
        ax.set_ylabel('Row (X)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def run_tests():
    print("\n=== TEST 1 : Sans obstacle ===")
    m1 = Maze(8, 12, (0, 0), (7, 11))
    assert m1.is_walkable(*m1.start)
    assert m1.is_walkable(*m1.goal)
    path1 = m1.solve()
    assert path1 is not None
    print("Longueur (4 directions):", len(path1) - 1)
    m1.display_with_path(path1)

    print("\n=== TEST 2 : Obstacles simples (solvable) ===")
    m2 = Maze(8, 12, (0, 0), (7, 11))
    m2.set_fixed_obstacles()
    assert m2.is_walkable(*m2.start)
    assert m2.is_walkable(*m2.goal)
    path2 = m2.solve()
    assert path2 is not None
    print("Longueur (4 directions):", len(path2) - 1)
    m2.display_with_path(path2)

    # Test avec diagonales
    path2_diag = m2.solve_with_diagonals()
    if path2_diag:
        dist = sum(m2.movement_cost(path2_diag[i][0], path2_diag[i][1],
                                      path2_diag[i+1][0], path2_diag[i+1][1])
                   for i in range(len(path2_diag) - 1))
        print("Longueur avec diagonales:", len(path2_diag) - 1, f"(distance: {dist:.2f})")
        m2.display_with_path(path2_diag)

    print("\n=== TEST 3 : Aucun chemin possible ===")
    m3 = Maze(8, 12, (0, 0), (7, 11))
    m3.add_obstacle(0, 1)
    m3.add_obstacle(1, 0)
    assert m3.is_walkable(*m3.start)
    assert m3.is_walkable(*m3.goal)
    path3 = m3.solve()
    assert path3 is None
    print("Aucun chemin trouvé (correct).")

    # Test visualisation
    print("\n=== TEST 4 : Visualisation graphique (matplotlib) ===")
    m4 = Maze(10, 15, (0, 0), (9, 14))
    m4.set_fixed_obstacles()
    path4 = m4.solve()
    if path4:
        print(f"Chemin trouvé avec {len(path4) - 1} étapes.")
        m4.visualize(path4, title="A* avec 4 directions")
        
        path4_diag = m4.solve_with_diagonals()
        if path4_diag:
            m4.visualize(path4_diag, title="A* avec 8 directions (diagonales)")

    # Test 5 : Comparaison A* vs Dijkstra
    print("\n=== TEST 5 : Comparaison A* vs Dijkstra ===")
    m5 = Maze(10, 15, (0, 0), (9, 14))
    m5.set_fixed_obstacles()
    m5.compare_algorithms()

    # Test 6 : Poids négatifs (récompenses)
    print("\n=== TEST 6 : Test avec poids négatifs (champs de récompense) ===")
    m6 = Maze(8, 12, (0, 0), (7, 11))
    m6.set_fixed_obstacles()
    
    # Créer des zones avec récompenses négatives (obstacles logiques / pénalités)
    negative_zones = {
        (3, 5): -10.0,  # Zone pénalisante
        (4, 4): -10.0,
        (5, 6): -10.0,
    }
    m6.init_rewards(step_penalty=-1.0, goal_reward=50.0, bonus_cells=negative_zones)
    
    print("Labyrinthe avec zones pénalisantes:")
    print("  Récompenses négatives (< -1) aux positions: (3,5), (4,4), (5,6)")
    m6.display()
    
    path6 = m6.solve()
    if path6:
        print(f"Chemin trouvé, longueur: {len(path6) - 1}")
        print(f"Positions du chemin: {path6}")
       
        m6.display_with_path(path6)


if __name__ == "__main__":
    run_tests()
