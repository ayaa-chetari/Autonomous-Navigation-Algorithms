import heapq
from typing import List, Tuple, Optional, Dict


class Maze:
    def __init__(self, height: int, width: int, start: Tuple[int, int], goal: Tuple[int, int]):
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal

        # Grille des obstacles (0 = libre, 1 = obstacle)
        self.grid: List[List[int]] = [[0 for _ in range(width)] for _ in range(height)]

        # Matrice des récompenses
        self.reward: List[List[float]] = [[0.0 for _ in range(width)] for _ in range(height)]

        if not self.in_bounds(*start) or not self.in_bounds(*goal):
            raise ValueError("Start ou Goal hors de la grille")

        # start et goal toujours libres
        self.grid[start[0]][start[1]] = 0
        self.grid[goal[0]][goal[1]] = 0

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.height and 0 <= y < self.width

    def is_walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.grid[x][y] == 0

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors: List[Tuple[int, int]] = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_walkable(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def add_obstacle(self, x: int, y: int) -> None:
        if (x, y) != self.start and (x, y) != self.goal and self.in_bounds(x, y):
            self.grid[x][y] = 1

    def remove_obstacle(self, x: int, y: int) -> None:
        if self.in_bounds(x, y):
            self.grid[x][y] = 0

    def set_reward(self, x: int, y: int, value: float) -> None:
        if self.in_bounds(x, y):
            self.reward[x][y] = value

    
    # Obstacles fixes 
   
    def set_fixed_obstacles(self) -> None:
   
        self.clear_obstacles()

        # Mur vertical colonne 3 avec DEUX ouvertures (ligne 1 et ligne 6)
        for i in range(self.height):
            if i not in (1, 6):
                self.add_obstacle(i, 3)

        # Mur vertical colonne 7 avec DEUX ouvertures (ligne 2 et ligne 5)
        for i in range(self.height):
            if i not in (2, 5):
                self.add_obstacle(i, 7)

        # Petit mur horizontal ligne 4, colonnes 8..10 avec DEUX ouvertures (9 et 10)
        for j in range(8, 11):
            if j not in (9, 10):
                self.add_obstacle(4, j)

        # Quelques obstacles internes pour créer des DÉTOURS / choix
       
        extra_blocks = [
            (1, 5), (2, 5),      
            (5, 9),             
            (6, 10),             
            (3, 9)             
        ]
        for x, y in extra_blocks:
            self.add_obstacle(x, y)

        # Sécurité : start/goal toujours libres
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]] = 0


  
    # Partie C : A* (solve)
   
    def heuristic_manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def solve(self) -> Optional[List[Tuple[int, int]]]:
        """
        Résout le labyrinthe avec A*.
        Retourne le chemin optimal [start, ..., goal] ou None si impossible.
        Coût de chaque déplacement = 1 (uniforme).
        """
        start = self.start
        goal = self.goal

        if start == goal:
            return [start]

        # Heap : (f, g, (x,y))
        open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set = set()

        f_start = self.heuristic_manhattan(start, goal)
        heapq.heappush(open_heap, (f_start, 0.0, start))

        while open_heap:
            f_current, g_current, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            cx, cy = current
            for neighbor in self.get_neighbors(cx, cy):
                if neighbor in closed_set:
                    continue

                tentative_g = g_current + 1.0

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_neighbor = tentative_g + self.heuristic_manhattan(neighbor, goal)
                    heapq.heappush(open_heap, (f_neighbor, tentative_g, neighbor))

        return None


    def init_rewards(
        self,
        step_penalty: float = -1.0,
        goal_reward: float = 100.0,
        bonus_cells: Optional[Dict[Tuple[int, int], float]] = None
    ) -> None:
        for i in range(self.height):
            for j in range(self.width):
                self.reward[i][j] = 0.0 if self.grid[i][j] == 1 else step_penalty

        if bonus_cells:
            for (x, y), val in bonus_cells.items():
                if self.in_bounds(x, y) and self.grid[x][y] == 0:
                    self.reward[x][y] = val

        gx, gy = self.goal
        self.reward[gx][gy] = goal_reward

    def clear_obstacles(self) -> None:
        for i in range(self.height):
            for j in range(self.width):
                self.grid[i][j] = 0

    def display(self) -> None:
        for i in range(self.height):
            row = ""
            for j in range(self.width):
                if (i, j) == self.start:
                    row += "S "
                elif (i, j) == self.goal:
                    row += "G "
                elif self.grid[i][j] == 1:
                    row += "# "
                else:
                    row += ". "
            print(row)

    def display_with_path(self, path: Optional[List[Tuple[int, int]]]) -> None:
        """Affiche le labyrinthe en montrant le chemin avec '*' (si path != None)."""
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


if __name__ == "__main__":
    m = Maze(8, 12, (0, 0), (7, 11))

    m.set_fixed_obstacles()
    m.init_rewards(step_penalty=-1.0, goal_reward=50.0, bonus_cells={(3, 3): 5.0})

    print("Labyrinthe :")
    m.display()

    path = m.solve()
    print("\nChemin A* :", path)
    if path is None:
        print("Aucun chemin trouvé.")
    else:
        print("Longueur du chemin:", len(path) - 1)

    print("\nLabyrinthe + chemin :")
    m.display_with_path(path)
