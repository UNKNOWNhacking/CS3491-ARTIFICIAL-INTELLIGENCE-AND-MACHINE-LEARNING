def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()
    
    # Mark the node as visited
    visited.add(node)
    print(node, end=" ")
    
    # Recur for all the vertices adjacent to this node
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)

# Example graph definition
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# Example usage:
print("\n\nDepth-First Search (Recursive):")
dfs_recursive(graph, 'A')
