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
