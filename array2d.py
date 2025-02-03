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

# https://leetcode.com/problems/rotate-image
def rotate(self, matrix: List[List[int]]) -> None:
    # swap across diagonal from top left to bottom right, then reverse rows
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):  # must avoid double swap by covering upper right (or lower left) of matrix
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    for row in matrix:
        for i in range(n // 2):
            row[i], row[n - 1 - i] = row[n - 1 - i], row[i]

# https://leetcode.com/problems/diagonal-traverse
def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
    xMax = len(mat[0]) - 1
    yMax = len(mat) - 1

    def offGrid(x, y):
        return x > xMax or x < 0 or y > yMax or y < 0

    up = True
    result = []
    x, y = 0, 0
    # if moving up & right and go off the grid, move right, if that's off grid too, instead move down
    # if moving down & left and go off the grid, move down, if that's off grid too, instead move right
    while True:
        result.append(mat[y][x])
        if up:
            if offGrid(x + 1, y - 1):  # try up and right
                if offGrid(x + 1, y):  # try right
                    if offGrid(x, y + 1):  # try down
                        break  # done
                    else:
                        up = False
                        x, y = x, y + 1
                else:
                    up = False
                    x, y = x + 1, y
            else:
                x, y = x + 1, y - 1
        else:
            if offGrid(x - 1, y + 1):  # try down and left
                if offGrid(x, y + 1):  # try down
                    if offGrid(x + 1, y):  # try right
                        break  # done
                    else:
                        up = True
                        x, y = x + 1, y
                else:
                    up = True
                    x, y = x, y + 1
            else:
                x, y = x - 1, y + 1
    return result

# https://leetcode.com/problems/toeplitz-matrix
def isToeplitzMatrixPretty(self, m: List[List[int]]) -> bool:
    for i in range(len(m) - 1):
        for j in range(len(m[0]) - 1):
            if m[i][j] != m[i + 1][j + 1]:
                return False
    return True
def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
    def toeplitzMatch(p, c):
        for i in range(1, len(c)):
            if c[i] != p[i-1]:
                return False
        return True
    # compare row by row; current row after 1st element must be the same as previous row offset by 1
    prevRow = matrix[0]
    for i in range(1, len(matrix)):
        curRow = matrix[i]
        if not toeplitzMatch(prevRow, curRow):
            return False
        prevRow = curRow
    return True
def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
    # start in bottom left and work the diagonals comparing pairs of numbers
    xMax = len(matrix[0])-1
    yMax = len(matrix)-1
    if xMax == 0 or yMax == 0:
        return True

    yStart = yMax-1
    xStart = 0
    prev = matrix[yStart][xStart]
    x = xStart+1
    y = yStart+1

    while True:
        if y > yMax or x > xMax:
            # reset to next diagonal
            xStart = xStart+1 if yStart == 0 else 0
            yStart = max(yStart-1, 0)
            prev = matrix[yStart][xStart]
            x = xStart+1
            y = yStart+1
        if x == xMax+1 and y == 1: # prev will be the upper right corner but current will be the next in diagonal
            break
        # print(f'checking (x,y) ({x},{y})')
        cur = matrix[y][x]
        if prev != cur:
            return False
        y += 1
        x += 1
    return True
