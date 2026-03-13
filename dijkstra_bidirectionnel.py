import heapq
from typing import List, Tuple, Dict, Optional, Any


def dijkstra_bidirectionnel(graph, start: Any, goal: Any) -> Tuple[Optional[List[Any]], float]:
    """
    Bidirectional Dijkstra's algorithm.
    
    Searches from both start and goal simultaneously until the two searches meet.
    This is generally faster than unidirectional Dijkstra.
    
    Args:
        graph: Graph object with get_neighbors() method
        start: Starting position
        goal: Goal position
    
    Returns:
        Tuple of (path, distance) where:
        - path: List of positions from start to goal, or None if no path exists
        - distance: Total distance of the path
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
        return None, float('inf')
    
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
    
    return path, best_path_length
