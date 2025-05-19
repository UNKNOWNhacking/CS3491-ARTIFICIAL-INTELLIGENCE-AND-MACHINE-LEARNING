import heapq

def a_star(graph, start, goal, h):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: h[start]}
    
    while open_list:
        _, current = heapq.heappop(open_list)
        
        if current == goal:
            return reconstruct_path(came_from, current)
        
        for neighbor, cost in graph[current].items():
            tentative_g = g_score[current] + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + h[neighbor]
                heapq.heappush(open_list, (f_score[neighbor], neighbor))
    
    return "Path not found"

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

# Example graph and heuristic
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'D': 1, 'E': 3},
    'C': {'E': 1},
    'D': {'F': 2},
    'E': {'F': 1},
    'F': {}
}

heuristic = {
    'A': 5, 'B': 4, 'C': 2,
    'D': 2, 'E': 1, 'F': 0
}

start_node = 'A'
goal_node = 'F'
# print(a_star(graph, start_node, goal_node, heuristic))

import heapq

def memory_bounded_a_star(graph, start, goal, h, memory_limit):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: h[start]}
    closed_list = set()
    
    while open_list:
        if len(open_list) > memory_limit:
            # Prune the least promising node
            open_list = sorted(open_list, key=lambda x: x[0])[:-1]
        
        _, current = heapq.heappop(open_list)
        
        if current == goal:
            return reconstruct_path(came_from, current)
        
        closed_list.add(current)
        
        for neighbor, cost in graph[current].items():
            tentative_g = g_score[current] + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                if neighbor not in closed_list:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + h[neighbor]
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
    
    return "Path not found"

# Example usage with memory limit
memory_limit = 4  # Limiting memory to store only 4 nodes at a time
print(memory_bounded_a_star(graph, start_node, goal_node, heuristic, memory_limit))
