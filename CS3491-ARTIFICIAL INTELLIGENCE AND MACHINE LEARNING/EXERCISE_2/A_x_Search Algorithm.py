import heapq

def a_star(graph, start, goal, h):
    # Priority queue for A* (min-heap)
    open_list = []
    heapq.heappush(open_list, (0, start))  # (f_score, node)
    
    # Dictionaries to store the actual cost and parent of each node
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    came_from = {}
    
    while open_list:
        # Get the node with the lowest f_score
        current_f, current = heapq.heappop(open_list)
        
        if current == goal:
            # Reconstruct and return the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return reversed path
        
        # Explore neighbors
        for neighbor, cost in graph[current]:
            tentative_g_score = g_score[current] + cost
            
            if tentative_g_score < g_score[neighbor]:
                # Update the best known path and g_score for neighbor
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + h(neighbor, goal)
                heapq.heappush(open_list, (f_score, neighbor))
    
    return None  # If no path is found

# Heuristic function (Manhattan distance for simplicity)
def heuristic(node, goal):
    x1, y1 = node
    x2, y2 = goal
    return abs(x1 - x2) + abs(y1 - y2)

# Example graph as an adjacency list
graph = {
    (0, 0): [((1, 0), 1), ((0, 1), 1)],
    (1, 0): [((0, 0), 1), ((1, 1), 1)],
    (0, 1): [((0, 0), 1), ((1, 1), 1)],
    (1, 1): [((1, 0), 1), ((0, 1), 1), ((2, 2), 1)],
    (2, 2): []
}

# Example usage
start = (0, 0)
goal = (2, 2)
print("A* Search Path:", a_star(graph, start, goal, heuristic))
