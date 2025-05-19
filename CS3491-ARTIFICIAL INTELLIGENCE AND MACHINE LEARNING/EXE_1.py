from collections import deque

def bfs(graph, start, goal):
    visited = set()
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        
        if node == goal:
            return f"Goal {goal} found"
        
        if node not in visited:
            visited.add(node)
            queue.extend(neighbor for neighbor in graph[node] if neighbor not in visited)
    
    return f"Goal {goal} not found"

# Graph as an adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

start_node = 'A'
goal_node = 'F'
print(bfs(graph, start_node, goal_node))



def dfs(graph, start, goal):
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        
        if node == goal:
            return f"Goal {goal} found"
        
        if node not in visited:
            visited.add(node)
            stack.extend(neighbor for neighbor in graph[node] if neighbor not in visited)
    
    return f"Goal {goal} not found"

# Graph as an adjacency list (same as above)
print(dfs(graph, start_node, goal_node))
