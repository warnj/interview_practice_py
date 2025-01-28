from collections import deque
from typing import List

# https://leetcode.com/problems/shortest-path-in-binary-matrix
# BFS
def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
    def validCell(cell):
        return cell[0] < len(grid[0]) and cell[1] < len(grid) and cell[0] >= 0 and cell[1] >= 0 and grid[cell[1]][cell[0]] == 0

    if grid[0][0] == 1 or grid[-1][-1] == 1:
        return -1
    wl = deque()
    wl.append((0,0,1)) # x, y, distance
    visited = set((0, 0))
    directions = [[-1,-1],[-1,0],[-1,1],[0,1],[0,-1],[1,-1],[1,0],[1,1]]

    while wl:
        x, y, dist = wl.popleft()
        if x == len(grid[0]) - 1 and y == len(grid) - 1:
            return dist

        # try visiting the adjacent cells
        for d in directions:
            nextCell = (x+d[0], y+d[1])
            if nextCell not in visited and validCell(nextCell):
                wl.append((nextCell[0], nextCell[1], dist+1))
                visited.add(nextCell) # important to mark visited as early as possible to pass time limit
    return -1

# https://leetcode.com/problems/accounts-merge/
def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
    # {email -> index in accounts} - problem is there could be multiple duplicates with 1 person and they may not find the same first index and it's inefficient to repeat the entire search to combine later
    # graph, treat emails as nodes and emails within a person are connected all connected (by single edge to the first email in the list), result is the connected components
    graph = {}  # {email -> [connected emails of the same person]}
    names = {}  # {email -> name} not complete
    for a in accounts:
        names[a[1]] = a[0]
        if a[1] not in graph:
            graph[a[1]] = []
        for j in range(2, len(a)):
            email = a[j]
            graph[a[1]].append(email)  # bidirectional edges
            if email in graph:
                graph[email].append(a[1])
            else:
                graph[email] = [a[1]]
    # explore connected components
    result = []
    visited = set()
    for email in graph:
        if email in visited:
            continue
        # explore this connected component dfs
        person = None
        emails = set()
        stack = [email]
        while stack:
            cur = stack.pop()
            visited.add(cur)
            if cur in names:
                person = names[cur]
            emails.add(cur)
            for child in graph[cur]:
                if child not in visited:
                    stack.append(child)
        emails = sorted(list(emails))
        result.append([person] + emails)
    return result
