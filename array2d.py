from typing import List


# https://leetcode.com/problems/number-of-islands
# O(n*m) time and space
def numIslands(self, grid: List[List[str]]) -> int:
    y = len(grid)
    x = len(grid[0])
    islands = 0
    bools = [[False for _ in range(x)] for _ in range(y)]
    for i in range(y):
        for j in range(x):
            if not bools[i][j] and grid[i][j] == '1':
                islands += 1
                self.__explore(grid, bools, i, j)
    return islands
def __explore(self, grid, bools, y, x):
    if y >= len(grid) or y < 0 or x >= len(grid[0]) or x < 0 or bools[y][x]:
        return
    if grid[y][x] == '1':
        bools[y][x] = True
        self.__explore(grid, bools, y + 1, x)
        self.__explore(grid, bools, y - 1, x)
        self.__explore(grid, bools, y, x + 1)
        self.__explore(grid, bools, y, x - 1)
