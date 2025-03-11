import sys
from collections import defaultdict, deque
from typing import List


# return the largest rhombus sum in a matrix of given size
    # example: size=2
    # [1,2,3,4]      2     3     7     8
    # [6,7,8,9]  -> 6 8   7 9   4 6   5 7  ->  max(21  25  25  27)  ->  27
    # [4,5,6,7]      5     6     8     7
    # [9,8,7,6]
def __rhombusSum(matrix, size, x, y):
    # print('summing for (x,y)=({},{})'.format(x,y))
    sum = 0
    xlo = x
    xhi = x
    for i in range(2*size-1):
        # print('i={} ylo={} yhi={}, xlo={}, xhi={}'.format(i, y, y+2*size-1, xlo, xhi))
        sum += matrix[y+i][xlo]
        if xlo != xhi:
            sum += matrix[y+i][xhi]
        if i < (2*size-1)//2:
            xlo -= 1
            xhi += 1
        else:
            xlo += 1
            xhi -= 1
    return sum
def getLargestRhombusSum(matrix, size):
    maxSum = -sys.maxsize
    # height/width of rhombus = 2*size-1
    for x in range(size-1, len(matrix[0])-size+1):
        for y in range(0, len(matrix)-(2*size-1)+1):
            # find (x,y) that is the top point of a rhombus
            curSum = __rhombusSum(matrix, size, x, y)
            # print('cursum={}'.format(curSum))
            maxSum = max(maxSum, curSum)
    return maxSum

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
    for i in range(1, len(matrix)):  # 1-2
        for j in range(i):  # 0-1
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    for i in range(len(matrix)):
        matrix[i].reverse()
    return matrix
    # (0,1) -> (1,0)
    # (0,2) -> (2,0)
    # (1,2) -> (2,1)
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
def findDiagonalOrderSoln(self, matrix: List[List[int]]) -> List[int]:
    if not matrix or not matrix[0]:
        return []
    N, M = len(matrix), len(matrix[0])
    row, column = 0, 0
    direction = 1
    result = []
    while row < N and column < M:
        # First and foremost, add the current element to
        # the result matrix.
        result.append(matrix[row][column])
        # Move along in the current diagonal depending upon
        # the current direction.[i, j] -> [i - 1, j + 1] if
        # going up and [i, j] -> [i + 1][j - 1] if going down.
        new_row = row + (-1 if direction == 1 else 1)
        new_column = column + (1 if direction == 1 else -1)
        # Checking if the next element in the diagonal is within the
        # bounds of the matrix or not. If it's not within the bounds,
        # we have to find the next head.
        if new_row < 0 or new_row == N or new_column < 0 or new_column == M:
            # If the current diagonal was going in the upwards
            # direction.
            if direction:
                # For an upwards going diagonal having [i, j] as its tail
                # If [i, j + 1] is within bounds, then it becomes
                # the next head. Otherwise, the element directly below
                # i.e. the element [i + 1, j] becomes the next head
                row += (column == M - 1)
                column += (column < M - 1)
            else:
                # For a downwards going diagonal having [i, j] as its tail
                # if [i + 1, j] is within bounds, then it becomes
                # the next head. Otherwise, the element directly below
                # i.e. the element [i, j + 1] becomes the next head
                column += (row == N - 1)
                row += (row < N - 1)
            # Flip the direction
            direction = 1 - direction
        else:
            row = new_row
            column = new_column
    return result

    # https://leetcode.com/problems/diagonal-traverse-ii
# O(n) time and O(sqrt(n)) space
def findDiagonalOrderBFS(self, nums: List[List[int]]) -> List[int]:
    queue = deque([(0, 0)])
    ans = []
    while queue:
        row, col = queue.popleft()
        ans.append(nums[row][col])
        if col == 0 and row + 1 < len(nums): # only look down on first col (x==0)
            queue.append((row + 1, col))
        if col + 1 < len(nums[row]): # always check right if it's valid
            queue.append((row, col + 1))
    return ans
def findDiagonalOrderBFS2(self, nums: List[List[int]]) -> List[int]:
    result = []
    q = deque([(0, 0)])
    while q:
        x, y = q.popleft()
        result.append(nums[y][x])
        if y + 1 < len(nums) and x < len(nums[y + 1]):  # down
            if not (q and q[-1] == (x, y + 1)):  # don't go down if it would be the same as previous element going right
                q.append((x, y + 1))
        if x + 1 < len(nums[y]):  # right
            q.append((x + 1, y))
    return result
# O(n) time and space
def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
    groups = defaultdict(list)
    for row in range(len(nums) - 1, -1, -1):
        for col in range(len(nums[row])):
            diagonal = row + col
            groups[diagonal].append(nums[row][col])
    ans = []
    curr = 0
    while curr in groups:
        ans.extend(groups[curr])
        curr += 1
    return ans
# O(nlogn) time and O(1) space
def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
    triples = []
    for y in range(len(nums)):
        for x in range(len(nums[y])):
            triples.append((x + y, x, nums[y][x]))
    triples.sort()
    return [t[2] for t in triples]
def findDiagonalOrderSlow(self, nums: List[List[int]]) -> List[int]:
    sx, sy = 0, 0
    maxX = max([len(row) for row in nums])
    result = []

    def traverse(x, y):
        while y >= 0 and x < maxX:
            if x < len(nums[y]):
                result.append(nums[y][x])
            y -= 1
            x += 1

    while sy < len(nums)-1:
        traverse(sx, sy)
        sy += 1
    while sx < maxX:
        traverse(sx, sy)
        sx += 1
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

# https://leetcode.com/problems/flood-fill
def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    start = image[sr][sc]

    def ff(x, y):
        if 0 <= x < len(image[0]) and 0 <= y < len(image) and image[y][x] == start:
            image[y][x] = color
            for dx, dy in directions:
                ff(x + dx, y + dy)

    if start != color:
        ff(sc, sr)
    return image

# https://leetcode.com/problems/shortest-bridge
def shortestBridge(self, grid: List[List[int]]) -> int:
    # explore 1st island, adding all water boundary points to a set
    # convert set to a queue and do multi-source bfs until hit 2nd island
    first = set()
    my = len(grid)
    mx = len(grid[0])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def convertFirst(x, y):
        grid[y][x] = 2  # mark as first island
        if x - 1 >= 0:
            if grid[y][x - 1] == 1:
                convertFirst(x - 1, y)  # explore more island
            elif grid[y][x - 1] == 0:
                first.add((x, y, 0))  # save island borders
        if x + 1 < mx:
            if grid[y][x + 1] == 1:
                convertFirst(x + 1, y)
            elif grid[y][x + 1] == 0:
                first.add((x, y, 0))
        if y - 1 >= 0:
            if grid[y - 1][x] == 1:
                convertFirst(x, y - 1)
            elif grid[y - 1][x] == 0:
                first.add((x, y, 0))
        if y + 1 < my:
            if grid[y + 1][x] == 1:
                convertFirst(x, y + 1)
            elif grid[y + 1][x] == 0:
                first.add((x, y, 0))

    def convertFirstIsland():
        for y in range(my):
            for x in range(mx):
                if grid[y][x] == 1:
                    convertFirst(x, y)
                    return

    def bfs(q):
        while q:
            cx, cy, d = q.popleft()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < mx and 0 <= ny < my:
                    if grid[ny][nx] == 1:
                        return d  # bridge len 1 less than the number of hops
                    if grid[ny][nx] == 0 or (
                            grid[ny][nx] < 0 and -grid[ny][nx] > d + 1):  # not explored or we can do better
                        grid[ny][nx] = -(d + 1)  # avoid further away bfs-es repeating this spot
                        q.append((nx, ny, d + 1))
        return float('inf')

    convertFirstIsland()

    result = float('inf')
    for start in first:
        result = min(result, bfs(deque([start])))
    return result
def shortestBridge2(self, grid: List[List[int]]) -> int:
    n = len(grid)
    first_x, first_y = -1, -1
    # Find any land cell, and we treat it as a cell of island A.
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                first_x, first_y = i, j
                break
    # Recursively check the neighboring land cell of current cell grid[x][y] and add all
    # land cells of island A to bfs_queue.
    def dfs(x, y):
        grid[x][y] = 2
        bfs_queue.append((x, y))
        for cur_x, cur_y in [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]:
            if (
                0 <= cur_x < n
                and 0 <= cur_y < n
                and grid[cur_x][cur_y] == 1
            ):
                dfs(cur_x, cur_y)
    # Add all land cells of island A to bfs_queue.
    bfs_queue = []
    dfs(first_x, first_y)
    distance = 0
    while bfs_queue:
        new_bfs = []
        for x, y in bfs_queue:
            for cur_x, cur_y in [
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            ]:
                if 0 <= cur_x < n and 0 <= cur_y < n:
                    if grid[cur_x][cur_y] == 1:
                        return distance
                    elif grid[cur_x][cur_y] == 0:
                        new_bfs.append((cur_x, cur_y))
                        grid[cur_x][cur_y] = -1
        # Once we finish one round without finding land cells of island B, we will
        # start the next round on all water cells that are 1 cell further away from
        # island A and increment the distance by 1.
        bfs_queue = new_bfs
        distance += 1
    return distance

# https://leetcode.com/problems/max-area-of-island
def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    my = len(grid)
    mx = len(grid[0])
    maxArea = 0
    for y in range(my):
        for x in range(mx):
            if grid[y][x] == 1:
                area = 0
                # explore the island
                grid[y][x] = 2  # mark visited
                stack = [(x, y)]
                while stack:
                    cx, cy = stack.pop()
                    area += 1
                    for dx, dy in directions:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < mx and 0 <= ny < my and grid[ny][nx] == 1:
                            grid[ny][nx] = 2
                            stack.append((nx, ny))
                maxArea = max(maxArea, area)
    return maxArea

# https://leetcode.com/problems/minesweeper
def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]

    def reveal(x, y):
        if 0 <= x < len(board[0]) and 0 <= y < len(board):
            tgt = board[y][x]
            if tgt == 'M':
                board[y][x] = 'X'
            elif tgt == 'E':
                # count mines adjacent
                mines = 0
                for dx, dy in directions:
                    if 0 <= x + dx < len(board[0]) and 0 <= y + dy < len(board) and (
                            board[y + dy][x + dx] == 'X' or board[y + dy][x + dx] == 'M'):
                        mines += 1
                # reveal the square
                if mines > 0:
                    board[y][x] = str(mines)
                else:
                    board[y][x] = 'B'
                    for dx, dy in directions:
                        reveal(x + dx, y + dy)

    reveal(click[1], click[0])
    return board

# https://leetcode.com/problems/making-a-large-island
# O(n*m) time and space, two pass solution
# on first pass give each island a unique number, change all cells in island to it
# also build a map from island id to size
# on 2nd pass try flipping each 0 to 1 and look around at bordering cells, take the 2 largest bordering islands as the new area
# return the max area, treat special case like no island or all island differently
def largestIslandBest(self, grid: List[List[int]]) -> int:
    island_sizes = {}
    island_id = 2 # on first pass give each island a unique number, change all cells in island to it

    # Step 1: Mark all islands and calculate their sizes
    for current_row in range(len(grid)):
        for current_column in range(len(grid[0])):
            if grid[current_row][current_column] == 1:
                island_sizes[island_id] = self.explore_island(
                    grid, island_id, current_row, current_column
                )
                island_id += 1

    # If there are no islands, return 1
    if not island_sizes:
        return 1

    # If the entire grid is one island, return its size or size +
    if len(island_sizes) == 1:
        island_id -= 1
        return (
            island_sizes[island_id]
            if island_sizes[island_id] == len(grid) * len(grid[0])
            else island_sizes[island_id] + 1
        )

    max_island_size = 1

    # Step 2: Try converting every 0 to 1 and calculate the resulting island size
    for current_row in range(len(grid)):
        for current_column in range(len(grid[0])):
            if grid[current_row][current_column] == 0:
                current_island_size = 1
                neighboring_islands = set()

                # Check down
                if (
                        current_row + 1 < len(grid)
                        and grid[current_row + 1][current_column] > 1
                ):
                    neighboring_islands.add(
                        grid[current_row + 1][current_column]
                    )

                # Check up
                if (
                        current_row - 1 >= 0
                        and grid[current_row - 1][current_column] > 1
                ):
                    neighboring_islands.add(
                        grid[current_row - 1][current_column]
                    )

                # Check right
                if (
                        current_column + 1 < len(grid[0])
                        and grid[current_row][current_column + 1] > 1
                ):
                    neighboring_islands.add(
                        grid[current_row][current_column + 1]
                    )

                # Check left
                if (
                        current_column - 1 >= 0
                        and grid[current_row][current_column - 1] > 1
                ):
                    neighboring_islands.add(
                        grid[current_row][current_column - 1]
                    )

                # Sum the sizes of all unique neighboring islands
                for island_id in neighboring_islands:
                    current_island_size += island_sizes[island_id]
                max_island_size = max(max_island_size, current_island_size)

    return max_island_size

def explore_island(
        self,
        grid: List[List[int]],
        island_id: int,
        current_row: int,
        current_column: int,
) -> int:
    if (
            current_row < 0
            or current_row >= len(grid)
            or current_column < 0
            or current_column >= len(grid[0])
            or grid[current_row][current_column] != 1
    ):
        return 0

    grid[current_row][current_column] = island_id

    return (
            1
            + self.explore_island(
        grid, island_id, current_row + 1, current_column
    )
            + self.explore_island(
        grid, island_id, current_row - 1, current_column
    )
            + self.explore_island(
        grid, island_id, current_row, current_column + 1
    )
            + self.explore_island(
        grid, island_id, current_row, current_column - 1
    )
    )
def largestIslandBetter(self, g: List[List[int]]) -> int:
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    my = len(g)
    mx = len(g[0])

    def onesArea(x, y, visited):
        boarder = set()  # set of points worth turning to land
        area = 0
        # explore the island
        visited.add((x, y))  # mark visited
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            area += 1
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < mx and 0 <= ny < my and (nx, ny) not in visited:
                    if g[ny][nx] == 1:
                        visited.add((nx, ny))
                        stack.append((nx, ny))
                    elif g[ny][nx] == 0:
                        boarder.add((nx, ny))
        return area, boarder

    def originalArea(x, y):
        visited = set()
        # check original island size, saving edge cells and visited coords
        original, boarder = onesArea(x, y, visited)
        originalVisits = set(visited)
        # try changing each boarder cell and add the extra island area we can explore after, save the max
        maxExtra = 0
        for bx, by in boarder:
            g[by][bx] = 1
            extra, _ = onesArea(bx, by, visited)
            maxExtra = max(maxExtra, extra)
            g[by][bx] = 0
        return original + maxExtra, originalVisits

    result = 0
    v = set()  # use to skip later cells in same island
    empty = True
    for y in range(my):
        for x in range(mx):
            if (x, y) not in v and g[y][x] == 1:
                empty = False
                a, v = originalArea(x, y)
                result = max(result, a)
            else:
                v.clear()
    return 1 if empty else result
def largestIslandCloseToWorkingSlow(self, g: List[List[int]]) -> int:
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    my = len(g)
    mx = len(g[0])

    def islandArea(x, y) -> int:
        boarder = set()  # set of points worth turning to land
        visited = set()
        area = 0
        # explore the island
        visited.add((x, y))  # mark visited
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            area += 1
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < mx and 0 <= ny < my:
                    if g[ny][nx] == 1:  # (nx,ny) not in visited and
                        visited.add((nx, ny))
                        stack.append((nx, ny))
                    elif g[ny][nx] == 0:
                        boarder.add((nx, ny))

        def explore(sx, sy):
            v2 = set()

        maxExtra = 0
        for bx, by in boarder:
            g[by][bx] = 1
            extra = 0

            explore(bx, by)

            maxExtra = max(maxExtra, extra)
            g[by][bx] = 0

        return area

    result = 0
    for y in range(my):
        for x in range(mx):
            if g[y][x] == 1:
                g[y][x] = 1
                result = max(result, islandArea(x, y))
                g[y][x] = 0
    return result

# https://leetcode.com/problems/spiral-matrix
def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
    result = []
    rows, columns = len(matrix), len(matrix[0])
    up = left = 0
    right = columns - 1
    down = rows - 1

    while len(result) < rows * columns:
        # Traverse from left to right.
        for col in range(left, right + 1):
            result.append(matrix[up][col])

        # Traverse downwards.
        for row in range(up + 1, down + 1):
            result.append(matrix[row][right])

        # Make sure we are now on a different row.
        if up != down:
            # Traverse from right to left.
            for col in range(right - 1, left - 1, -1):
                result.append(matrix[down][col])

        # Make sure we are now on a different column.
        if left != right:
            # Traverse upwards.
            for row in range(down - 1, up, -1):
                result.append(matrix[row][left])

        left += 1
        right -= 1
        up += 1
        down -= 1
    return result

# https://leetcode.com/problems/word-search-ii/
# Time complexity: O(M(4⋅ 3 ^ (L−1))), where M is the number of cells in the board and L is the maximum length of words.
# space O(n) where n is total number of letters in dict
def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
    WORD_KEY = "$"
    trie = {}
    for word in words:
        node = trie
        for letter in word:
            # retrieve the next node; If not found, create a empty node.
            node = node.setdefault(letter, {})
        # mark the existence of a word in trie node
        node[WORD_KEY] = word

    rowNum = len(board)
    colNum = len(board[0])
    matchedWords = []

    def backtracking(row, col, parent):
        letter = board[row][col]
        currNode = parent[letter]

        # check if we find a match of word
        word_match = currNode.pop(WORD_KEY, False)
        if word_match:
            # also we removed the matched word to avoid duplicates,
            #   as well as avoiding using set() for results.
            matchedWords.append(word_match)

        # Before the EXPLORATION, mark the cell as visited
        board[row][col] = "#"

        # Explore the neighbors in 4 directions, i.e. up, right, down, left
        for rowOffset, colOffset in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            newRow, newCol = row + rowOffset, col + colOffset
            if (
                newRow < 0
                or newRow >= rowNum
                or newCol < 0
                or newCol >= colNum
            ):
                continue
            if not board[newRow][newCol] in currNode:
                continue
            backtracking(newRow, newCol, currNode)

        # End of EXPLORATION, we restore the cell
        board[row][col] = letter

        # Optimization: incrementally remove the matched leaf node in Trie.
        if not currNode:
            parent.pop(letter)

    for row in range(rowNum):
        for col in range(colNum):
            # starting from each of the cells
            if board[row][col] in trie:
                backtracking(row, col, trie)
    return matchedWords

# https://leetcode.com/problems/longest-increasing-path-in-a-matrix
'''
// DFS + Memoization Solution, O(mn) time and space
public class Solution {
    private static final int[][] dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    private int m, n;

    public int longestIncreasingPath(int[][] matrix) {
        if (matrix.length == 0) return 0;
        m = matrix.length; n = matrix[0].length;
        int[][] cache = new int[m][n];
        int ans = 0;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                ans = Math.max(ans, dfs(matrix, i, j, cache));
        return ans;
    }

    private int dfs(int[][] matrix, int i, int j, int[][] cache) {
        if (cache[i][j] != 0) return cache[i][j];
        for (int[] d : dirs) {
            int x = i + d[0], y = j + d[1];
            if (0 <= x && x < m && 0 <= y && y < n && matrix[x][y] > matrix[i][j])
                cache[i][j] = Math.max(cache[i][j], dfs(matrix, x, y, cache));
        }
        return ++cache[i][j];
    }
}'''