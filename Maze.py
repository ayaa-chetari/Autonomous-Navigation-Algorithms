import heapq
from typing import List, Tuple, Optional, Dict


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

    def display_with_path(self, path):
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


def run_tests():
    print("\n=== TEST 1 : Sans obstacle ===")
    m1 = Maze(8, 12, (0, 0), (7, 11))
    assert m1.is_walkable(*m1.start)
    assert m1.is_walkable(*m1.goal)
    path1 = m1.solve()
    assert path1 is not None
    print("Longueur:", len(path1) - 1)
    m1.display_with_path(path1)

    print("\n=== TEST 2 : Obstacles simples (solvable) ===")
    m2 = Maze(8, 12, (0, 0), (7, 11))
    m2.set_fixed_obstacles()
    assert m2.is_walkable(*m2.start)
    assert m2.is_walkable(*m2.goal)
    path2 = m2.solve()
    assert path2 is not None
    print("Longueur:", len(path2) - 1)
    m2.display_with_path(path2)

    print("\n=== TEST 3 : Aucun chemin possible ===")
    m3 = Maze(8, 12, (0, 0), (7, 11))
    m3.add_obstacle(0, 1)
    m3.add_obstacle(1, 0)
    assert m3.is_walkable(*m3.start)
    assert m3.is_walkable(*m3.goal)
    path3 = m3.solve()
    assert path3 is None
    print("Aucun chemin trouvé (correct).")



if __name__ == "__main__":
    run_tests()
