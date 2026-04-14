"""
TP3 — Robotique : Résoudre un labyrinthe avec Q-learning
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import networkx as nx
from collections import defaultdict


class MazeEnv:
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ACTION_NAMES = ['↑', '↓', '←', '→']

    def __init__(self, maze_matrix):
        self.maze = maze_matrix
        self.rows = len(maze_matrix)
        self.cols = len(maze_matrix[0])
        self.start = None
        self.goal = None
        for i in range(self.rows):
            for j in range(self.cols):
                if self.maze[i][j] == 'S':
                    self.start = (i, j)
                elif self.maze[i][j] == 'G':
                    self.goal = (i, j)
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action_idx):
        di, dj = self.ACTIONS[action_idx]
        ni, nj = self.state[0] + di, self.state[1] + dj
        if ni < 0 or ni >= self.rows or nj < 0 or nj >= self.cols:
            return self.state, -100, False
        cell = self.maze[ni][nj]
        if cell == 1:
            return self.state, -100, False
        if cell == 'G':
            self.state = (ni, nj)
            return self.state, +100, True
        self.state = (ni, nj)
        return self.state, -1, False




class GraphEnv:
    """
    Environnement RL basé sur un graphe NetworkX.
    - États : nœuds du graphe
    - Actions : indices des voisins (variable selon l'état)
    - Récompenses : +100 (goal), -1 (déplacement)
    """

    def __init__(self, G, start, goal):
        self.G = G
        self.start = start
        self.goal = goal
        self.state = start

    def reset(self):
        self.state = self.start
        return self.state

    def get_neighbors(self, state):
        """Retourne la liste des voisins d'un nœud."""
        return list(self.G.neighbors(state))

    def step(self, next_node):
        """Se déplacer vers un nœud voisin."""
        if next_node == self.goal:
            self.state = next_node
            return self.state, +100, True
        self.state = next_node
        return self.state, -1, False



def q_learning_maze(env, alpha=0.1, gamma=0.95, epsilon0=1.0, epsilon_min=0.01,
                    epsilon_decay=0.995, n_episodes=1000, max_steps=200):
    Q = defaultdict(lambda: np.zeros(4))
    epsilon = epsilon0
    rewards_per_episode = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while state != env.goal and steps < max_steps:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done = env.step(action)
            best_next = np.max(Q[next_state])
            Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])
            state = next_state
            total_reward += reward
            steps += 1
            if done:
                break

        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        rewards_per_episode.append(total_reward)

    return dict(Q), rewards_per_episode



def q_learning_graph(env, alpha=0.1, gamma=0.95, epsilon0=1.0, epsilon_min=0.01,
                     epsilon_decay=0.995, n_episodes=1000, max_steps=500):
    """
    Q-learning sur un graphe. Q[state] est un dict {voisin: valeur}.
    """
    Q = defaultdict(lambda: defaultdict(float))
    epsilon = epsilon0
    rewards_per_episode = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while state != env.goal and steps < max_steps:
            neighbors = env.get_neighbors(state)
            if not neighbors:
                break

            # Epsilon-greedy
            if random.random() < epsilon:
                next_node = random.choice(neighbors)
            else:
                # Meilleure action connue
                q_vals = {n: Q[state][n] for n in neighbors}
                max_q = max(q_vals.values())
                best = [n for n, v in q_vals.items() if v == max_q]
                next_node = random.choice(best)

            # Exécuter l'action
            next_state, reward, done = env.step(next_node)

            # max Q(s', a') sur les voisins de s'
            next_neighbors = env.get_neighbors(next_state)
            if next_neighbors:
                best_next = max(Q[next_state][nn] for nn in next_neighbors)
            else:
                best_next = 0

            # Mise à jour Q(s, a)
            Q[state][next_node] += alpha * (reward + gamma * best_next - Q[state][next_node])

            state = next_state
            total_reward += reward
            steps += 1
            if done:
                break

        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        rewards_per_episode.append(total_reward)

    return dict(Q), rewards_per_episode


def extract_path_graph(Q, env, max_steps=500):
    """Extraire le chemin optimal depuis la Q-table sur un graphe."""
    path = [env.start]
    state = env.start
    visited = set()

    for _ in range(max_steps):
        if state == env.goal:
            break
        if state in visited:
            break
        visited.add(state)

        neighbors = env.get_neighbors(state)
        if not neighbors or state not in Q:
            break

        q_vals = {n: Q[state][n] for n in neighbors}
        next_node = max(q_vals, key=q_vals.get)
        state = next_node
        path.append(state)

    return path



def extract_policy(Q, env):
    policy = {}
    for state, q_values in Q.items():
        policy[state] = np.argmax(q_values)
    return policy


def extract_path_maze(policy, env, max_steps=200):
    path = [env.start]
    state = env.start
    visited = set()
    for _ in range(max_steps):
        if state == env.goal:
            break
        if state in visited:
            break
        visited.add(state)
        if state not in policy:
            break
        action = policy[state]
        di, dj = MazeEnv.ACTIONS[action]
        ns = (state[0] + di, state[1] + dj)
        if (0 <= ns[0] < env.rows and 0 <= ns[1] < env.cols
                and env.maze[ns[0]][ns[1]] != 1):
            state = ns
            path.append(state)
        else:
            break
    return path



def plot_convergence(rewards, title="Convergence Q-learning", window=50):
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, alpha=0.25, color='steelblue')
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(rewards)), smoothed, color='orangered', linewidth=2)
    plt.xlabel('Épisode')
    plt.ylabel('Récompense cumulée')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()




def plot_qtable(Q, env, title="Q-table (valeur max par état)"):
    grid = np.full((env.rows, env.cols), np.nan)
    for (i, j), qv in Q.items():
        if env.maze[i][j] != 1:
            grid[i][j] = np.max(qv)
    plt.figure(figsize=(env.cols + 1, env.rows + 1))
    plt.imshow(grid, cmap='YlOrRd', origin='upper')
    for i in range(env.rows):
        for j in range(env.cols):
            if not np.isnan(grid[i][j]):
                plt.text(j, i, f'{grid[i][j]:.0f}', ha='center', va='center', fontsize=9)
    plt.colorbar(label='max Q(s,a)')
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontweight='bold')
    plt.tight_layout()


def plot_graph_with_path(G, path, start, goal, title="Graphe — Chemin Q-learning"):
    """Visualiser le graphe avec le chemin trouvé."""
    plt.figure(figsize=(10, 8))
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        pos = nx.spring_layout(G, seed=42)

    # Couleurs des nœuds
    path_set = set(path)
    node_colors = []
    for n in G.nodes():
        if n == start:
            node_colors.append('#4a90d9')
        elif n == goal:
            node_colors.append('#4caf50')
        elif n in path_set:
            node_colors.append('#ffb74d')
        else:
            node_colors.append('#ddd')

    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=20)

    # Dessiner le chemin en surbrillance
    if len(path) > 1:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='orangered',
                               width=2.5, alpha=0.9)

    # Labels start/goal
    nx.draw_networkx_labels(G, pos, labels={start: 'S', goal: 'G'},
                            font_size=12, font_weight='bold', font_color='white')

    plt.title(title, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()



if __name__ == "__main__":

    print("=== Question 1 : Labyrinthe 4×4 ===")

    maze_tp3 = [
        ['S', 0,  1,  1 ],
        [ 1,  0,  0,  0 ],
        [ 1,  0,  1,  0 ],
        [ 1,  0,  0, 'G'],
    ]

    env = MazeEnv(maze_tp3)
    Q, rewards = q_learning_maze(env, n_episodes=1000)
    policy = extract_policy(Q, env)
    path = extract_path_maze(policy, env)
    print(f"Chemin : {path}  ({len(path)-1} pas)")

    plot_convergence(rewards, "Convergence — Labyrinthe 4×4")
    plt.savefig('convergence_4x4.png', dpi=150)


    plt.savefig('maze_policy_4x4.png', dpi=150)

    plot_qtable(Q, env, "Q-table — Labyrinthe 4×4")
    plt.savefig('qtable_4x4.png', dpi=150)

    
    print("\n=== Test : Labyrinthe 8×8 ===")

    maze_8x8 = [
        ['S', 0,  1,  0,  0,  0,  1,  0 ],
        [ 0,  0,  1,  0,  1,  0,  0,  0 ],
        [ 1,  0,  0,  0,  1,  1,  1,  0 ],
        [ 1,  1,  1,  0,  0,  0,  0,  0 ],
        [ 0,  0,  0,  0,  1,  1,  0,  1 ],
        [ 0,  1,  1,  0,  0,  0,  0,  1 ],
        [ 0,  0,  1,  1,  1,  1,  0,  0 ],
        [ 1,  0,  0,  0,  0,  1,  0, 'G'],
    ]

    env8 = MazeEnv(maze_8x8)
    Q8, rewards8 = q_learning_maze(env8, n_episodes=3000, max_steps=500)
    policy8 = extract_policy(Q8, env8)
    path8 = extract_path_maze(policy8, env8)
    print(f"Chemin : {path8}  ({len(path8)-1} pas)")

    plot_convergence(rewards8, "Convergence — Labyrinthe 8×8")
    plt.savefig('convergence_8x8.png', dpi=150)

    plt.savefig('maze_policy_8x8.png', dpi=150)


    graphs = {
        "Random Geometric": nx.random_geometric_graph(2000, 0.05, seed=42),
        "Erdos-Renyi":      nx.erdos_renyi_graph(1000, 0.01, seed=42),
        "Barabasi-Albert":  nx.barabasi_albert_graph(1000, 5, seed=42),
    }

    all_rewards = {}

    for name, G in graphs.items():
        print(f"\n  {name} ({G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes)")

        # Choisir start et goal éloignés dans la plus grande composante connexe
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc).copy()
        nodes_list = list(subG.nodes())

        # Pour le graphe géométrique, prendre les nœuds les plus éloignés
        if 'pos' in nx.get_node_attributes(G, 'pos'):
            pos = nx.get_node_attributes(subG, 'pos')
            # Trouver les 2 nœuds les plus éloignés
            max_dist = 0
            start, goal = nodes_list[0], nodes_list[-1]
            sample = random.sample(nodes_list, min(100, len(nodes_list)))
            for u in sample:
                for v in sample:
                    d = np.sqrt((pos[u][0]-pos[v][0])**2 + (pos[u][1]-pos[v][1])**2)
                    if d > max_dist:
                        max_dist = d
                        start, goal = u, v
        else:
            # Pour les autres graphes, prendre deux nœuds éloignés en hops
            start = random.choice(nodes_list)
            # BFS pour trouver le nœud le plus éloigné
            lengths = nx.single_source_shortest_path_length(subG, start)
            goal = max(lengths, key=lengths.get)

        print(f"  Start: {start}, Goal: {goal}")
        print(f"  Plus court chemin (NetworkX): {nx.shortest_path_length(subG, start, goal)} hops")

        env_g = GraphEnv(subG, start, goal)
        Q_g, rew_g = q_learning_graph(
            env_g, alpha=0.1, gamma=0.95, epsilon0=1.0,
            epsilon_min=0.01, epsilon_decay=0.998, n_episodes=3000, max_steps=500
        )
        path_g = extract_path_graph(Q_g, env_g, max_steps=500)
        reached = path_g[-1] == goal if path_g else False
        all_rewards[name] = rew_g
        print(f"  Chemin Q-learning: {len(path_g)-1} hops, goal atteint: {reached}")

        # Visualiser le graphe avec le chemin
        plot_graph_with_path(subG, path_g, start, goal,
                             title=f"{name} — Chemin Q-learning ({len(path_g)-1} hops)")
        plt.savefig(f'graph_{name.lower().replace(" ", "_").replace("é", "e")}.png', dpi=150)

    # ─── Comparaison des convergences ───
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, rew) in zip(axes, all_rewards.items()):
        ax.plot(rew, alpha=0.25, color='steelblue')
        w = 50
        if len(rew) >= w:
            sm = np.convolve(rew, np.ones(w) / w, mode='valid')
            ax.plot(range(w - 1, len(rew)), sm, color='orangered', linewidth=2)
        ax.set_title(name)
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Récompense')
        ax.grid(True, alpha=0.3)
    fig.suptitle("Comparaison de la convergence sur différents graphes", fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparaison_graphes.png', dpi=150)

    plt.close('all')
    print("\nTerminé.")