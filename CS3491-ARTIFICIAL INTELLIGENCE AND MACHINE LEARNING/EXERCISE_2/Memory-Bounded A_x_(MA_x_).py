import heapq

class Node:
    def __init__(self, state, g, h, parent=None):
        self.state = state
        self.g = g  # Cost from start to node
        self.h = h  # Heuristic estimate
        self.f = g + h  # Estimated total cost
        self.parent = parent  # To reconstruct the path

    def __lt__(self, other):
        return self.f < other.f

def memory_bounded_a_star(graph, start, goal, h, memory_limit):
    # Priority queue (open list) for nodes to explore
    open_list = []
    heapq.heappush(open_list, Node(start, 0, h(start, goal)))

    best_solution = None  # Track the best path

    while open_list:
        # Trim open_list if memory limit is exceeded
        if len(open_list) > memory_limit:
            open_list = heapq.nsmallest(memory_limit, open_list)  # Keep the best nodes

        # Get the node with the lowest f-score
        current = heapq.heappop(open_list)

        if current.state == goal:
            # Goal reached: reconstruct the path
            path = []
            while current:
                path.append(current.state)
                current = current.parent
            return path[::-1]  # Return reversed path (from start to goal)

        # Explore neighbors
        for neighbor, cost in graph[current.state]:
            g = current.g + cost  # Update the g-score (actual cost to the neighbor)
            f = g + h(neighbor, goal)  # Total estimated cost f(n) = g(n) + h(n)
            heapq.heappush(open_list, Node(neighbor, g, h(neighbor, goal), current))

        # Track the best solution so far based on f-score
        if not best_solution or current.f < best_solution.f:
            best_solution = current

    # If no solution is found, return the best solution found so far
    return None if best_solution is None else reconstruct_path(best_solution)

def reconstruct_path(node):
    """Reconstruct the path from start to goal by following parent nodes."""
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]  # Return the path from start to goal

# Heuristic function (Manhattan distance for this example)
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

# Example usage with memory limit
start = (0, 0)
goal = (2, 2)
memory_limit = 5

print("Memory-Bounded A* Search Path:", memory_bounded_a_star(graph, start, goal, heuristic, memory_limit))
