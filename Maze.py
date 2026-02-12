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


    def set_fixed_obstacles(self) -> None:
     
        self.clear_obstacles()

        for i in range(self.height):
            if i != 1:  
                self.add_obstacle(i, 3)

        
        for i in range(self.height):
            if i != 5:  
                self.add_obstacle(i, 7)

        
        for j in range(8, 11):
            if j != 9:  
                self.add_obstacle(4, j)

      
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]] = 0

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


if __name__ == "__main__":
    m = Maze(8, 12, (0, 0), (7, 11))

   
    m.set_fixed_obstacles()

    m.init_rewards(step_penalty=-1.0, goal_reward=50.0, bonus_cells={(3, 3): 5.0})
    m.display()

   