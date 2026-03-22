import heapq
from typing import List, Tuple, Dict, Optional, Any


def dijkstra_bidirectionnel(graph, start: Any, goal: Any) -> Tuple[Optional[List[Any]], float, int]:
    """
    Bidirectional Dijkstra's algorithm.
    
    Searches from both start and goal simultaneously until the two searches meet.
    This is generally faster than unidirectional Dijkstra.
    
    Args:
        graph: Graph object with get_neighbors() method
        start: Starting position
        goal: Goal position
    
    Returns:
        Tuple of (path, distance, nodes_explored) where:
        - path: List of positions from start to goal, or None if no path exists
        - distance: Total distance of the path
        - nodes_explored: Total number of nodes visited during the search
    """
    
    # Forward search (from start)
    forward_heap: List[Tuple[float, Tuple[int, int]]] = [(0, start)]
    forward_dist: Dict[Tuple[int, int], float] = {start: 0}
    forward_parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    forward_visited: Dict[Tuple[int, int], float] = {}
    
    # Backward search (from goal)
    backward_heap: List[Tuple[float, Tuple[int, int]]] = [(0, goal)]
    backward_dist: Dict[Tuple[int, int], float] = {goal: 0}
    backward_parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    backward_visited: Dict[Tuple[int, int], float] = {}
    
    # To track the meeting point
    best_path_length = float('inf')
    meeting_point = None
    
    while forward_heap and backward_heap:
        # Expand from forward direction
        if forward_heap:
            forward_dist_curr, current_forward = heapq.heappop(forward_heap)
            
            if current_forward in forward_visited:
                continue
            
            forward_visited[current_forward] = forward_dist_curr
            
            # Check if this node has been visited by backward search
            if current_forward in backward_visited:
                path_length = forward_dist_curr + backward_visited[current_forward]
                if path_length < best_path_length:
                    best_path_length = path_length
                    meeting_point = current_forward
            
            # Expand neighbors
            for neighbor in graph.get_neighbors(current_forward):
                new_dist = forward_dist_curr + 1
                
                if neighbor not in forward_dist or new_dist < forward_dist[neighbor]:
                    forward_dist[neighbor] = new_dist
                    forward_parent[neighbor] = current_forward
                    heapq.heappush(forward_heap, (new_dist, neighbor))
        
        # Expand from backward direction
        if backward_heap:
            backward_dist_curr, current_backward = heapq.heappop(backward_heap)
            
            if current_backward in backward_visited:
                continue
            
            backward_visited[current_backward] = backward_dist_curr
            
            # Check if this node has been visited by forward search
            if current_backward in forward_visited:
                path_length = forward_visited[current_backward] + backward_dist_curr
                if path_length < best_path_length:
                    best_path_length = path_length
                    meeting_point = current_backward
            
            # Expand neighbors
            for neighbor in graph.get_neighbors(current_backward):
                new_dist = backward_dist_curr + 1
                
                if neighbor not in backward_dist or new_dist < backward_dist[neighbor]:
                    backward_dist[neighbor] = new_dist
                    backward_parent[neighbor] = current_backward
                    heapq.heappush(backward_heap, (new_dist, neighbor))
        
        # Early termination if we found a path and both searches have explored
        # nodes that are farther than our best path
        if (meeting_point is not None and 
            forward_heap and backward_heap and
            forward_heap[0][0] + backward_heap[0][0] >= best_path_length):
            break
    
    # Reconstruct path if found
    if meeting_point is None:
        return None, float('inf'), len(forward_visited) + len(backward_visited)
    
    # Build forward path (from start to meeting point)
    path_forward = []
    current = meeting_point
    while current != start:
        path_forward.append(current)
        current = forward_parent[current]
    path_forward.append(start)
    path_forward.reverse()
    
    # Build backward path (from meeting point to goal)
    path_backward = []
    current = meeting_point
    while current != goal:
        path_backward.append(current)
        current = backward_parent[current]
    path_backward.append(goal)
    
    # Combine paths
    path = path_forward + path_backward[1:]
    
    return path, best_path_length, len(forward_visited) + len(backward_visited)


def dijkstra_classique(graph, start: Any, goal: Any) -> Tuple[Optional[List[Any]], float, int]:
    """
    Classic unidirectional Dijkstra's algorithm.
    
    Searches from start node until reaching goal node.
    
    Args:
        graph: Graph object with get_neighbors() method
        start: Starting position
        goal: Goal position
    
    Returns:
        Tuple of (path, distance, nodes_explored) where:
        - path: List of positions from start to goal, or None if no path exists
        - distance: Total distance of the path
        - nodes_explored: Total number of nodes visited during the search
    """
    heap: List[Tuple[float, Any]] = [(0, start)]
    dist: Dict[Any, float] = {start: 0}
    parent: Dict[Any, Optional[Any]] = {start: None}
    visited: Dict[Any, float] = {}
    
    while heap:
        curr_dist, current = heapq.heappop(heap)
        
        if current in visited:
            continue
        
        visited[current] = curr_dist
        
        # Check if we reached the goal
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, curr_dist, len(visited)
        
        # Expand neighbors
        for neighbor in graph.get_neighbors(current):
            new_dist = curr_dist + 1
            
            if neighbor not in dist or new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                parent[neighbor] = current
                heapq.heappush(heap, (new_dist, neighbor))
    
    # No path found
    return None, float('inf'), len(visited)


def astar_bidirectionnel(graph, start: Any, goal: Any, positions: Dict[Any, Tuple[float, float]]) -> Tuple[Optional[List[Any]], float, int]:
    """
    Bidirectional A* algorithm using Euclidean distance as heuristic.
    
    Searches from both start and goal simultaneously using heuristic guidance.
    This is generally faster than bidirectional Dijkstra when good heuristics are available.
    
    Args:
        graph: Graph object with get_neighbors() method
        start: Starting position
        goal: Goal position
        positions: Dict mapping node -> (x, y) coordinates for heuristic calculation
    
    Returns:
        Tuple of (path, distance, nodes_explored) where:
        - path: List of positions from start to goal, or None if no path exists
        - distance: Total distance of the path
        - nodes_explored: Total number of nodes visited during the search
    """
    
    def heuristic(node_a: Any, node_b: Any) -> float:
        """Euclidean distance heuristic between two nodes"""
        if node_a not in positions or node_b not in positions:
            return 0  # Zero heuristic if positions not available
        try:
            pos_a = positions[node_a]
            pos_b = positions[node_b]
            # Ensure positions are tuples with 2 elements
            if not isinstance(pos_a, (tuple, list)) or not isinstance(pos_b, (tuple, list)):
                return 0
            if len(pos_a) < 2 or len(pos_b) < 2:
                return 0
            x1, y1 = float(pos_a[0]), float(pos_a[1])
            x2, y2 = float(pos_b[0]), float(pos_b[1])
            return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        except (TypeError, ValueError, KeyError):
            return 0  # Return zero heuristic on any error
    
    # Forward search (from start)
    forward_heap: List[Tuple[float, Any]] = [(heuristic(start, goal), start)]
    forward_dist: Dict[Any, float] = {start: 0}
    forward_parent: Dict[Any, Any] = {}
    forward_visited: Dict[Any, float] = {}
    
    # Backward search (from goal)
    backward_heap: List[Tuple[float, Any]] = [(heuristic(goal, start), goal)]
    backward_dist: Dict[Any, float] = {goal: 0}
    backward_parent: Dict[Any, Any] = {}
    backward_visited: Dict[Any, float] = {}
    
    # To track the meeting point
    best_path_length = float('inf')
    meeting_point = None
    
    while forward_heap and backward_heap:
        # Expand from forward direction
        if forward_heap:
            forward_f, current_forward = heapq.heappop(forward_heap)
            forward_dist_curr = forward_dist[current_forward]
            
            if current_forward in forward_visited:
                pass
            else:
                forward_visited[current_forward] = forward_dist_curr
                
                # Check if this node has been visited by backward search
                if current_forward in backward_visited:
                    path_length = forward_dist_curr + backward_visited[current_forward]
                    if path_length < best_path_length:
                        best_path_length = path_length
                        meeting_point = current_forward
                
                # Expand neighbors
                for neighbor in graph.get_neighbors(current_forward):
                    new_dist = forward_dist_curr + 1
                    
                    if neighbor not in forward_dist or new_dist < forward_dist[neighbor]:
                        forward_dist[neighbor] = new_dist
                        forward_parent[neighbor] = current_forward
                        f_value = new_dist + heuristic(neighbor, goal)
                        heapq.heappush(forward_heap, (f_value, neighbor))
        
        # Expand from backward direction
        if backward_heap:
            backward_f, current_backward = heapq.heappop(backward_heap)
            backward_dist_curr = backward_dist[current_backward]
            
            if current_backward in backward_visited:
                pass
            else:
                backward_visited[current_backward] = backward_dist_curr
                
                # Check if this node has been visited by forward search
                if current_backward in forward_visited:
                    path_length = forward_visited[current_backward] + backward_dist_curr
                    if path_length < best_path_length:
                        best_path_length = path_length
                        meeting_point = current_backward
                
                # Expand neighbors
                for neighbor in graph.get_neighbors(current_backward):
                    new_dist = backward_dist_curr + 1
                    
                    if neighbor not in backward_dist or new_dist < backward_dist[neighbor]:
                        backward_dist[neighbor] = new_dist
                        backward_parent[neighbor] = current_backward
                        f_value = new_dist + heuristic(neighbor, start)
                        heapq.heappush(backward_heap, (f_value, neighbor))
        
        # Early termination if we found a path and both searches have explored
        # nodes that are farther than our best path
        if (meeting_point is not None and 
            forward_heap and backward_heap and
            forward_heap[0][0] + backward_heap[0][0] >= best_path_length):
            break
    
    # Reconstruct path if found
    if meeting_point is None:
        return None, float('inf'), len(forward_visited) + len(backward_visited)
    
    # Build forward path (from start to meeting point)
    path_forward = []
    current = meeting_point
    while current != start:
        path_forward.append(current)
        current = forward_parent[current]
    path_forward.append(start)
    path_forward.reverse()
    
    # Build backward path (from meeting point to goal)
    path_backward = []
    current = meeting_point
    while current != goal:
        path_backward.append(current)
        current = backward_parent[current]
    path_backward.append(goal)
    
    # Combine paths
    path = path_forward + path_backward[1:]
    
    return path, best_path_length, len(forward_visited) + len(backward_visited)