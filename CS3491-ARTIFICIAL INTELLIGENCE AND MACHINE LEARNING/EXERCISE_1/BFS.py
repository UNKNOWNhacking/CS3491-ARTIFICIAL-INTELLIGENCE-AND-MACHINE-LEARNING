from collections import deque

def bfs(graph, start_node):
    # Create a queue for BFS
    queue = deque([start_node])
    # Set to keep track of visited nodes
    visited = set([start_node])
    
    while queue:
        # Dequeue a node from the queue
        node = queue.popleft()
        print(node, end=" ")
        
        # Get all adjacent vertices of the dequeued node
        for neighbor in graph[node]:
            if neighbor not in visited:
                # Mark neighbor as visited and enqueue it
                visited.add(neighbor)
                queue.append(neighbor)

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print("Breadth-First Search:")
bfs(graph, 'A')
