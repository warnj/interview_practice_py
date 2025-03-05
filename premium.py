import collections
import heapq
import math
from bisect import bisect_left
from collections import defaultdict, deque, Counter
from typing import List, Optional


class Node:
    def __init__(self, val=0, left=None, right=None, parent=None):
        self.val = val
        self.left = left
        self.right = right
        self.parent = parent

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class NestedInteger:
   def isInteger(self):
       pass
   def getInteger(self):
       pass
   def getList(self):
       pass

# https://leetcode.com/problems/meeting-rooms
# Given an array of meeting time intervals where intervals[i] = [starti, endi], determine if a person could attend all meetings.
def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
    intervals.sort(key=lambda x: x[0])
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]:
            return False
    return True

# https://leetcode.com/problems/meeting-rooms-ii
# Given an array of meeting time intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required.
# Example 1: [[0,30],[5,10],[15,20]] -> 2
# Example 2: [[7,10],[2,4]] -> 1
# O(nlogn) time and O(n) space
def minMeetingRooms(self, intervals: List[List[int]]) -> int:
    # maxiumum number of overlaps
    # [1,5],[1,10],[2,3],[5,8]      -> 3
    # [1,5],[1,10],[2,4],[3,5]      -> 4
    # [1,10],[2,4],[6,8]            -> 2
    # [4,19],[6,13],[11,20],[13,17] -> 3    first and last could overlap and middle ones not -> min heap
    intervals.sort(key=lambda x: x[0])
    maxOverlaps = 0
    heap = []  # stores endpoints of currently overlapping intervals
    for n in intervals:
        # current interval does not overlap with everything in heap, remove from heap until it does overlap with everything
        while heap and heap[0] <= n[0]:
            heapq.heappop(heap)
        heapq.heappush(heap, n[1])
        maxOverlaps = max(maxOverlaps, len(heap))
    return maxOverlaps

# https://leetcode.com/problems/logger-rate-limiter
# easy - prevent log messages from being printed more frequently than 1 every 10s
class Logger:
    def __init__(self):
        self.msgs = {} # msg -> time
    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        if message in self.msgs:
            ts = self.msgs[message]
            if timestamp - ts < 10:
                return False
        self.msgs[message] = timestamp
        return True

# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii
# Given two nodes of a binary tree p and q, return their lowest common ancestor (LCA).
#   same as: https://leetcode.com/problems/intersection-of-two-linked-lists
# O(n+m) time and O(1) space    -  lists share a same-length common ending
# travel to the end, then reset as the starting point of other list, when the 2nd pointer gets assigned to start of shorter list the 1st pointer is offset the right amount for corresponding nodes in the longer list
def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
    p1, p2 = p, q
    while p1 != p2:
        p1 = p1.parent if p1.parent else q
        p2 = p2.parent if p2.parent else p
    return p1
# O(n+m) time and space - save every node
def lowestCommonAncestorBad(self, p: 'Node', q: 'Node') -> 'Node':
    parents = set()
    cur = p
    while cur:
        parents.add(cur)
        cur = cur.parent
    cur = q
    while cur:
        if cur in parents:
            return cur
        cur = cur.parent
    return None

# https://leetcode.com/problems/binary-tree-vertical-order-traversal
#    same as https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree
#    but with ties by level order instead of sorted order
# O(n) time and O(n) space
def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    # xmin & xmax
    # dict: x -> [values in this col]   work top to bottom & left to right
    if not root:
        return []
    cols = defaultdict(list)
    xmin, xmax = 0, 0

    # level-order traversal
    wl = deque([(root, 0)])  # (node, x-val)
    while wl:
        size = len(wl)
        for _ in range(size):
            cur, x = wl.popleft()
            cols[x].append(cur.val)
            xmin = min(xmin, x)
            xmax = max(xmax, x)

            if cur.left:
                wl.append((cur.left, x - 1))
            if cur.right:
                wl.append((cur.right, x + 1))
    result = []
    for i in range(xmin, xmax + 1):
        result.append(cols[i]) # know that it is an unbroken range
    return result

# https://leetcode.com/problems/design-tic-tac-toe/
# Assume the following rules are for the tic-tac-toe game on an n x n board between two players:
# A move is guaranteed to be valid and is placed on an empty block.
# Once a winning condition is reached, no more moves are allowed.
# A player who succeeds in placing n of their marks in a horizontal, vertical, or diagonal row wins the game.
# Implement the TicTacToe class:
# TicTacToe(int n) Initializes the object the size of the board n.
# int move(int row, int col, int player) Indicates that the player with id player plays at the cell (row, col) of the board. The move is guaranteed to be a valid move, and the two players alternate in making moves. Return
# 0 if there is no winner after the move,
# 1 if player 1 is the winner after the move, or
# 2 if player 2 is the winner after the move.
class TicTacToe:
    def __init__(self, n: int):
        # count number of 1s & 2s in each row, col, and diagonal
        self.rows = [0] * n
        self.cols = [0] * n
        self.diags = 0
        self.alts = 0 # alternate diagonal

    def move(self, row: int, col: int, player: int) -> int:
        cur = -1 if player == 2 else 1  # go up/down based on the current player, if abs == n then current player wins

        self.rows[row] += cur
        if abs(self.rows[row]) == len(self.rows):
            return player
        self.cols[col] += cur
        if abs(self.cols[col]) == len(self.cols):
            return player
        # incr diagonals
        if row == col:
            self.diags += cur
            if abs(self.diags) == len(self.cols):
                return player
        if row == len(self.cols) - 1 - col or col == len(self.cols) - 1 - row:  # are both checks needed?
            self.alts += cur
            if abs(self.alts) == len(self.cols):
                return player
        return 0

class TicTacToeEasy:
    def __init__(self, n: int):
        self.board = [[0]*n for _ in range(n)]

    def move(self, row: int, col: int, player: int) -> int:
        # row is y val
        self.board[row][col] = player
        n = len(self.board)
        # count number in a row from here looking up, down, left, right, diagonal

        # easy diagonal
        if row == col:
            for i in range(1, n):
                if not self.board[i][i] or self.board[i][i] != self.board[i-1][i-1]:
                    break
                if i == n-1:
                    return player
        # odd diagonal
        if row == n - 1 - col or col == n - 1 - row: # are both checks needed?
            for i in range(1, n): # bottom left to top right
                if not self.board[n-1-i][i] or self.board[n-1-i][i] != self.board[n-i][i-1]:
                    break
                if i == n-1:
                    return player
        # row
        r = self.board[row]
        for i in range(1, n):
            if not r[i] or r[i] != r[i-1]:
                break
            if i == n-1:
                return player
        # col
        for i in range(1, n):
            if not self.board[i][col] or self.board[i][col] != self.board[i-1][col]:
                break
            if i == n-1:
                return player
        return 0

# https://leetcode.com/problems/longest-repeating-substring
# Given a string s, return the length of the longest repeating substrings. If no repeating substring exists, return 0.
# Input: s = "abcd"  Output: 0  Explanation: There is no repeating substring.
# Input: s = "abbaba" Output: 2  Explanation: The longest repeating substrings are "ab" and "ba", each of which occurs twice.
# Input: s = "aabcaabdaab" Output: 3  Explanation: The longest repeating substring is "aab", which occurs 3 times.
# O(n^2 *logn) time and O(n^2) space
def longestRepeatingSubstring(self, s: str) -> int:
    start, end = 1, len(s) - 1
    while start <= end:
        mid = (start + end) // 2
        # Check if there's a repeating substring of length mid
        if self._has_repeating_substring(s, mid):
            start = mid + 1  # Try longer substrings
        else:
            end = mid - 1  # Try shorter substrings
    return start - 1  # just found the lowest length without repeating substrs so return 1 less than that
def _has_repeating_substring(self, s: str, length: int) -> bool:
    seen = set()
    for i in range(len(s) - length + 1):  # i is starting locations for substrs
        substring = s[i: i + length]
        if substring in seen:
            return True
        seen.add(substring)
    return False
# O(n^2) time and space
# store the length of the longest repeating substring that ends at i
def longestRepeatingSubstringDP(s: str) -> int:
    length = len(s)
    # each cell (i, j) records length of longest common suffix (the end portion of a substring) of substrings ending at i and j
    dp = [[0] * (length + 1) for _ in range(length + 1)]  # longer than needed by 1 to simplify the code needed to start from 0
    max_length = 0
    # fill table by comparing each character in the string with every other character after it. If S[i] == S[j] and i != j, it means the characters match
    # and we can extend the length of the common suffix we found previously by 1.
    for i in range(1, length + 1):
        for j in range(i + 1, length + 1): # don't compare same character with itself so start at i+1
            if s[i - 1] == s[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_length = max(max_length, dp[i][j])
    return max_length  # max value in the table (most of the table will be 0)
# O(n^3) time and O(n^2) space
def longestRepeatingSubstringBruteBad(self, s: str) -> int:
    subs = defaultdict(int)  # substring -> count of it
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            subs[s[i:j]] += 1
    maxSize = 0
    for k, v in subs.items():
        if v > 1 and len(k) > maxSize:
            maxSize = len(k)
    return maxSize

# https://leetcode.com/problems/number-of-distinct-islands
# You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.
# An island is considered to be the same as another if and only if one island can be translated (and not rotated or reflected) to equal the other.
# Return the number of distinct islands.
# O(nm) time and O(nm) space (if you're not allowed to modify given grid, otherwise O(1))
def numDistinctIslands(self, grid: List[List[int]]) -> int:
    islands = set()
    # represent an island as a tuple of sorted (x,y) pairs offset from the topmost left point of island
    yLen = len(grid)
    xLen = len(grid[0])

    def explore(x, xOg, y, yOg, island):
        island.append((x - xOg, y - yOg))
        grid[y][x] = -1  # mark as visited

        if x + 1 < xLen and grid[y][x + 1] == 1:
            explore(x + 1, xOg, y, yOg, island)
        if x - 1 >= 0 and grid[y][x - 1] == 1:
            explore(x - 1, xOg, y, yOg, island)
        if y + 1 < yLen and grid[y + 1][x] == 1:
            explore(x, xOg, y + 1, yOg, island)
        if y - 1 >= 0 and grid[y - 1][x] == 1:
            explore(x, xOg, y - 1, yOg, island)

    for y in range(yLen):
        for x in range(xLen):
            if grid[y][x] == 1:
                island = []
                explore(x, x, y, y, island)
                # island.sort() # don't need this since all islands are discovered in the same relative order
                islands.add(tuple(island))
    return len(islands)

# https://leetcode.com/problems/number-of-distinct-islands-ii
# You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.
# An island is considered to be the same as another if they have the same shape, or have the same shape after rotation (90, 180, or 270 degrees only) or reflection (left/right direction or up/down direction).
# Return the number of distinct islands.
#   easy in theory - find a way to standardize the islands that ignores orientation, but challenging to implement
def numDistinctIslands2(self, grid):
    def dfs(x, y, coords):
        if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]) or grid[x][y] == 0:
            return
        grid[x][y] = 0
        coords.append((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            dfs(x + dx, y + dy, coords)

    def canonical(coords):
        shapes = []
        for i, j in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            # Reflection
            shape = sorted([(x * i, y * j) for x, y in coords])
            shape = [(x - shape[0][0], y - shape[0][1]) for x, y in shape]
            shapes.append(shape)

            # Rotations
            shape = sorted([(y * i, x * j) for x, y in coords])
            shape = [(x - shape[0][0], y - shape[0][1]) for x, y in shape]
            shapes.append(shape)

        return min(shapes)

    distinct_islands = set()
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            if grid[x][y] == 1:
                coords = []
                dfs(x, y, coords)
                distinct_islands.add(tuple(canonical(coords)))

    return len(distinct_islands)

# https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters
# Given a string s and an integer k, return the length of the longest substring of s that contains at most k distinct characters.
# Input: s = "eceba", k = 2  Output: 3  Explanation: The substring is "ece" with length 3.
# Input: s = "aa", k = 1  Output: 2  Explanation: The substring is "aa" with length 2.
# O(n) time and O(k) space
def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
    if k == 0: return 0
    distinct = 0
    counts = defaultdict(int) # counts of chars in the window
    lo, hi = 0, 0
    result = 1
    while hi < len(s):
        c = s[hi]
        counts[c] += 1
        if counts[c] == 1:  # adding a new char to window
            distinct += 1

        while distinct > k:
            # shorten from left
            c2 = s[lo]
            counts[c2] -= 1
            if counts[c2] == 0:  # not a distinct char in window anymore
                distinct -= 1
            lo += 1

        # window is valid and hi & lo are inclusive
        result = max(result, hi - lo + 1)
        hi += 1
    return result
# O(n) time and space (assuming >=n unique chars exist) - faster avg time than above since it doesn't shrink the window below max_size
def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
    max_size = 0
    counter = Counter()
    for right in range(len(s)):
        counter[s[right]] += 1
        if len(counter) <= k:
            max_size += 1
        else:
            counter[s[right - max_size]] -= 1
            if counter[s[right - max_size]] == 0:
                del counter[s[right - max_size]]
    return max_size

# You are given a nested list of integers nestedList. Each element is either an integer or a list whose elements may also be integers or other lists.
# The depth of an integer is the number of lists that it is inside of. For example, the nested list [1,[2,2],[[3],2],1] has each integer's value set to its depth.
# Return the sum of each integer in nestedList multiplied by its depth.
# Input: nestedList = [[1,1],2,[1,1]]   Output: 10   Explanation: Four 1's at depth 2, one 2 at depth 1. 1*2 + 1*2 + 2*1 + 1*2 + 1*2 = 10.
# Input: nestedList = [1,[4,[6]]]   Output: 27   Explanation: One 1 at depth 1, one 4 at depth 2, and one 6 at depth 3. 1*1 + 4*2 + 6*3 = 27.
# https://leetcode.com/problems/nested-list-weight-sum
# O(n) time and space
def depthSum(self, nestedList: List[NestedInteger]) -> int:
    s = 0
    def processInt(nestedInt, depth):
        nonlocal s
        l = nestedInt.getList()
        if l:
            for i in l:
                processInt(i, depth + 1)
        elif nestedInt.isInteger():
            s += nestedInt.getInteger() * depth
    for nInt in nestedList:
        processInt(nInt, 1)
    return s

# https://leetcode.com/problems/alien-dictionary
# There is a new alien language that uses the English alphabet. However, the order of the letters is unknown to you.
# You are given a list of strings words from the alien language's dictionary. Now it is claimed that the strings in words are
# sorted lexicographically by the rules of this new language.
# If this claim is incorrect, and the given arrangement of string in words cannot correspond to any order of letters, return "".
# Otherwise, return a string of the unique letters in the new alien language sorted in lexicographically increasing order by the new language's rules. If there are multiple solutions, return any of them.
# Input: words = ["wrt","wrf","er","ett","rftt"]   Output: "wertf"
# Input: words = ["z","x"]   Output: "zx"
# Input: words = ["z","x","z"]   Output: ""    order is invalid
def alienOrder(self, words: List[str]) -> str:
    # graph like pre-requisites and topo sort
    # {letter -> set(letters lexicographically larger)}
    # how to handle single-letter words?
    graph = defaultdict(set)
    indegree = {}
    indegree[words[0][0]] = 0  # need to track chars that may not be in graph (but should be) and have no incomming edges
    for i in range(1, len(words)):
        prevW = words[i - 1]
        curW = words[i]
        shortest = min(len(prevW), len(curW))
        j = 0
        while j < shortest:
            c = prevW[j]
            c2 = curW[j]
            if c != c2:  # c -> c2
                children = graph[c]
                if c2 not in children:  # add a new child node
                    children.add(c2)
                    indegree[c2] = indegree.get(c2, 0) + 1
                break  # first non-matching char determined the word ordering so can't infer anything about following chars
            j += 1
        if j == len(curW) and j < len(prevW):  # 2nd word shorter than first but otherwise identical
            return ''
    for w in words:  # ensure terminating nodes are in the graph as keys and all starting nodes have an indegree
        for c in w:
            if c not in graph:
                graph[c] = set()
            if c not in indegree:
                indegree[c] = 0
    # topo sort
    wl = deque()
    for char, count in indegree.items():
        if count == 0:
            wl.append(char)
    visited = set()
    result = []
    while wl:
        cur = wl.popleft()
        result.append(cur)
        visited.add(cur)
        children = graph.get(cur)
        if children:
            for child in children:
                indegree[child] -= 1
                if indegree[child] == 0 and child not in visited:
                    wl.append(child)

    if len(visited) != len(graph):  # a cycle was found
        return ''
    return ''.join(result)

def alienOrderBFS(self, words: List[str]) -> str:
    # Step 0: create data structures + the in_degree of each unique letter to 0.
    adj_list = defaultdict(set)
    in_degree = Counter({c: 0 for word in words for c in word})

    # Step 1: We need to populate adj_list and in_degree.
    # For each pair of adjacent words...
    for first_word, second_word in zip(words, words[1:]):
        for c, d in zip(first_word, second_word):
            if c != d:
                if d not in adj_list[c]:
                    adj_list[c].add(d)
                    in_degree[d] += 1
                break
        else:  # Check that second word isn't a prefix of first word.
            if len(second_word) < len(first_word):
                return ""

    # Step 2: We need to repeatedly pick off nodes with an indegree of 0.
    output = []
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    while queue:
        c = queue.popleft()
        output.append(c)
        for d in adj_list[c]:
            in_degree[d] -= 1
            if in_degree[d] == 0:
                queue.append(d)

    # If not all letters are in output, that means there was a cycle and so
    # no valid ordering. Return "" as per the problem description.
    if len(output) < len(in_degree):
        return ""
    # Otherwise, convert the ordering we found into a string and return it.
    return "".join(output)

def alienOrderDFS(self, words: List[str]) -> str:
    # Step 0: Put all unique letters into the adj list.
    reverse_adj_list = {c: [] for word in words for c in word}

    # Step 1: Find all edges and put them in reverse_adj_list.
    for first_word, second_word in zip(words, words[1:]):
        for c, d in zip(first_word, second_word):
            if c != d:
                reverse_adj_list[d].append(c)
                break
        else:  # Check that second word isn't a prefix of first word.
            if len(second_word) < len(first_word):
                return ""

    # Step 2: Depth-first search.
    seen = {}  # False = grey, True = black.
    output = []

    def visit(node):  # Return True iff there are no cycles.
        if node in seen:
            return seen[node]  # If this node was grey (False), a cycle was detected.
        seen[node] = False  # Mark node as grey.
        for next_node in reverse_adj_list[node]:
            result = visit(next_node)
            if not result:
                return False  # Cycle was detected lower down.
        seen[node] = True  # Mark node as black.
        output.append(node)
        return True

    if not all(visit(node) for node in reverse_adj_list):
        return ""
    return "".join(output)

# https://leetcode.com/problems/moving-average-from-data-stream
# Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.
# Implement the MovingAverage class:
# MovingAverage(int size) Initializes the object with the size of the window size.
# double next(int val) Returns the moving average of the last size values of the stream.
# Example 1:
# Input
# ["MovingAverage", "next", "next", "next", "next"]
# [[3], [1], [10], [3], [5]]
# Output
# [null, 1.0, 5.5, 4.66667, 6.0]
# Explanation
# MovingAverage movingAverage = new MovingAverage(3);
# movingAverage.next(1); // return 1.0 = 1 / 1
# movingAverage.next(10); // return 5.5 = (1 + 10) / 2
# movingAverage.next(3); // return 4.66667 = (1 + 10 + 3) / 3
# movingAverage.next(5); // return 6.0 = (10 + 3 + 5) / 3
class MovingAverage:
    def __init__(self, size: int):
        self.vals = deque()
        self.size = size
        self.sum = 0
    def next(self, val: int) -> float:
        if self.size == len(self.vals):
            self.sum -= self.vals.popleft()
        self.vals.append(val)
        self.sum += val
        return self.sum / len(self.vals)
class MovingAverageBad:
    def __init__(self, size: int):
        self.vals = deque()
        self.size = size
    def next(self, val: int) -> float:
        if self.size == len(self.vals):
            self.vals.popleft()
        self.vals.append(val)
        return sum(self.vals) / len(self.vals)

# https://leetcode.com/problems/closest-binary-search-tree-value
# Given the root of a binary search tree and a target value, return the value in the BST that is closest to the target. If there are multiple answers, print the smallest.
def closestValue(self, root: Optional[TreeNode], target: float) -> int:
    cur = root
    closest = root.val
    while cur:
        diff = abs(target - closest) - abs(target - cur.val)
        if diff > 0 or (diff == 0 and cur.val < closest):
            closest = cur.val
        if cur.left and cur.right:
            if target < cur.val:
                cur = cur.left
            else:
                cur = cur.right
        elif cur.left:
            cur = cur.left
        else:
            cur = cur.right
    return closest
def closestValuePretty(self, root: TreeNode, target: float) -> int:
    closest = root.val
    while root:
        closest = min(root.val, closest, key=lambda x: (abs(target - x), x))
        root = root.left if target < root.val else root.right
    return closest

# https://leetcode.com/problems/valid-word-abbreviation/
# A string can be abbreviated by replacing any number of non-adjacent, non-empty substrings with their lengths. The lengths should not have leading zeros.
# For example, a string such as "substitution" could be abbreviated as (but not limited to):
# "s10n" ("s ubstitutio n")
# "sub4u4" ("sub stit u tion")
# "12" ("substitution")
# "su3i1u2on" ("su bst i t u ti on")
# "substitution" (no substrings replaced)
# The following are not valid abbreviations:
# "s55n" ("s ubsti tutio n", the replaced substrings are adjacent)
# "s010n" (has leading zeros)
# "s0ubstitution" (replaces an empty substring)
# Given a string word and an abbreviation abbr, return whether the string matches the given abbreviation.
# A substring is a contiguous non-empty sequence of characters within a string.
# Input: word = "internationalization", abbr = "i12iz4n"  Output: true
# Explanation: The word "internationalization" can be abbreviated as "i12iz4n" ("i nternational iz atio n").
# Input: word = "apple", abbr = "a2e"   Output: false
# Explanation: The word "apple" cannot be abbreviated as "a2e".
def validWordAbbreviation(self, word: str, abbr: str) -> bool:
    digit = ''
    wi = 0
    for i in range(len(abbr)):
        c = abbr[i]
        if c.isdigit():
            if not digit and c == '0':
                return False  # leading zero
            digit += c
        else:
            if digit:  # digit just ended
                skip = int(digit)
                wi += skip
                digit = ''
            if wi >= len(word) or word[wi] != abbr[i]:
                return False
            wi += 1
        if wi > len(word):
            return False
    if digit:  # ended on a digit
        skip = int(digit)
        wi += skip
    return wi == len(word)

# https://leetcode.com/problems/split-bst
# Given the root of a binary search tree (BST) and an integer target, split the tree into two subtrees where the first subtree has nodes that are all smaller or equal to the target value, while the second subtree has all nodes that are greater than the target value. It is not necessarily the case that the tree contains a node with the value target.
# Additionally, most of the structure of the original tree should remain. Formally, for any child c with parent p in the original tree, if they are both in the same subtree after the split, then node c should still have the parent p.
# Return an array of the two roots of the two subtrees in order.
# note: this is surprisingly challenging
# O(h) time and space where h is the tree height
def splitBST(self, root: Optional[TreeNode], target: int) -> List[Optional[TreeNode]]:
    if not root:
        return [None, None]
    if root.val > target: # root (current node) belongs on second tree, but some of its left children may not, recursively split left child
        left = self.splitBST(root.left, target)
        # Attach the right part of the split to root's left subtree
        root.left = left[1]
        return [left[0], root]
    else: # root (current node) belongs on first tree, but some of its right children may not, recursively split right child
        right = self.splitBST(root.right, target)
        # Attach the left part of the split to root's right subtree
        root.right = right[0]
        return [root, right[1]]
# O(h) time and space
def splitBST(
    self, root: Optional[TreeNode], target: int
) -> List[Optional[TreeNode]]:
    # List to store the two split trees
    ans = [None, None]
    # If root is None, return the empty list
    if not root:
        return ans
    # Stack to traverse the tree and find the split point
    stack = []
    # Find the node with the value closest to the target
    while root:
        stack.append(root)
        if root.val > target:
            root = root.left
        else:
            root = root.right
    # Process nodes in reverse order from the stack to perform the split
    while stack:
        curr = stack.pop()
        if curr.val > target:
            # Assign current node's left child to the subtree
            # containing nodes greater than the target
            curr.left = ans[1]
            # current node becomes the new root of this subtree
            ans[1] = curr
        else:
            # Assign current node's right child to the subtree
            # containing nodes smaller than the target
            curr.right = ans[0]
            # current node becomes the new root of this subtree
            ans[0] = curr
    return ans
# O(n) time and O(1) space
def splitBST(self, root: TreeNode, target: int) -> list[TreeNode]:
    # Create dummy nodes to hold the split tree parts
    dummy_sm = TreeNode(0)
    cur_sm = dummy_sm
    dummy_lg = TreeNode(0)
    cur_lg = dummy_lg
    # Start traversal from the root
    current = root
    next_node = None
    while current is not None:
        if current.val <= target:
            # Attach the current node to the tree
            # with values less than or equal to the target
            cur_sm.right = current
            cur_sm = current
            # Move to the right subtree
            next_node = current.right
            # Clear the right pointer of current node
            cur_sm.right = None
            current = next_node
        else:
            # Attach the current node to the tree
            # with values greather to the target
            cur_lg.left = current
            cur_lg = current
            # Move to the left subtree
            next_node = current.left
            # Clear the left pointer of current node
            cur_lg.left = None
            current = next_node
    # Return the split parts as a list
    return [dummy_sm.right, dummy_lg.left]

# https://leetcode.com/problems/insert-into-a-sorted-circular-linked-list/
# Given a Circular Linked List node, which is sorted in non-descending order, write a function to insert a value insertVal into the list such that it remains a sorted circular list. The given node can be a reference to any single node in the list and may not necessarily be the smallest value in the circular list.
# If there are multiple suitable places for insertion, you may choose any place to insert the new value. After the insertion, the circular list should remain sorted.
# If the list is empty (i.e., the given node is null), you should create a new single circular list and return the reference to that single node. Otherwise, you should return the originally given node.
def insert(self, head: 'Optional[Node]', insertVal: int) -> 'Node':
    if not head:  # create a one-item list
        new = Node(insertVal)
        new.next = new
        return new
    if head.next == head:  # create a 2-item list inserting after head
        new = Node(insertVal, head)
        head.next = new
        return head

    cur = head
    while True:
        if cur.val <= insertVal <= cur.next.val:
            break
        if cur.val == insertVal:  # put equivalent value in first ok spot
            break
        if cur.next.val < cur.val:
            # currently at the max
            if insertVal < cur.next.val or insertVal >= cur.val:  # greater than the max or less than min, insert in next position
                break
        if cur.next == head:  # have looped across the entire list
            break
        cur = cur.next

    cur.next = Node(insertVal, cur.next)
    return head
def insertUgly(self, head: 'Optional[Node]', insertVal: int) -> 'Node':
    if not head:  # create a one-item list
        new = Node(insertVal)
        new.next = new
        return new
    if head.next == head:  # create a 2-item list inserting after head
        new = Node(insertVal, head)
        head.next = new
        return head

    cur = head
    foundMax = False
    while cur.next.val <= insertVal or not foundMax:
        if cur.val <= insertVal <= cur.next.val:
            break
        if cur.next.val == insertVal:  # put equivalent value in first ok spot
            break
        if cur.next.val < cur.val:
            # currently at the max
            foundMax = True
            if insertVal < cur.next.val or insertVal >= cur.val:  # greater than the max or less than mid, insert in next position
                break
        if cur.next == head:  # have looped across the entire list
            if cur.next.val == cur.val:  # edge case where entire loop is the same
                break
            foundMax = True  # sometimes we need to go back around and insert if elements after first are > insert i.e. [1,3,5] insert 2
        cur = cur.next

    cur.next = Node(insertVal, cur.next)
    return head
def insertNaiveIncorrect(self, head: 'Optional[Node]', insertVal: int) -> 'Node':
    if not head:  # create a one-item list
        new = Node(insertVal)
        new.next = new
        return new

    cur = head
    if insertVal <= cur.val:  # insert at front
        # end is the last node
        end = head.next
        while end.next != head:
            end = end.next

        new = Node(insertVal, head)
        end.next = new
        return new

    while cur.next and cur.next.val < insertVal:
        cur = cur.next

    # cur.next is either off the end or
    if not cur.next:
        # inserting at the end
        cur.next = Node(insertVal, head)
    else:
        cur.next = Node(insertVal, cur.next)
    return head

# https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list
# Convert a Binary Search Tree to a sorted Circular Doubly-Linked List in place.
# You can think of the left and right pointers as synonymous to the predecessor and successor pointers in a doubly-linked list. For a circular doubly linked list, the predecessor of the first element is the last element, and the successor of the last element is the first element.
# We want to do the transformation in place. After the transformation, the left pointer of the tree node should point to its predecessor, and the right pointer should point to its successor. You should return the pointer to the smallest element of the linked list.
def treeToDoublyList(self, root: 'Optional[Node]') -> 'Optional[Node]':
    if not root:
        return None
    smallest, prev = None, None

    # in-order traversal saving the previous node to connect the current one with
    def convert(cur):
        nonlocal smallest, prev
        if cur:
            convert(cur.left)
            if prev:
                prev.right = cur
                cur.left = prev
            else:
                smallest = cur
            prev = cur
            convert(cur.right)

    convert(root)
    prev.right = smallest
    smallest.left = prev
    return smallest
# overly complicated first attempt that works
def treeToDoublyListFirst(self, root: 'Optional[Node]') -> 'Optional[Node]':
    if not root:
        return None
    smallest = root
    largest = root

    def convert(cur):
        nonlocal smallest, largest
        if cur:
            if cur.val < smallest.val: smallest = cur
            elif cur.val > largest.val: largest = cur

            prev = convert(cur.left)
            if prev:
                while prev.right: # prev could now be in the middle of a smaller linked list, go to the end
                    prev = prev.right
                prev.right = cur
                cur.left = prev

            post = convert(cur.right)
            if post:
                while post.left:  # post could be in middle of a smaller linkedlist, go to its start
                    post = post.left
                post.left = cur
                cur.right = post
        return cur

    convert(root)
    largest.right = smallest
    smallest.left = largest
    return smallest

# https://leetcode.com/problems/dot-product-of-two-sparse-vectors
# Given two sparse vectors, compute their dot product.
# Implement class SparseVector:
# SparseVector(nums) Initializes the object with the vector nums
# dotProduct(vec) Compute the dot product between the instance of SparseVector and vec
# A sparse vector is a vector that has mostly zero values, you should store the sparse vector efficiently and compute the dot product between two SparseVector.
# Follow up: What if only one of the vectors is sparse?
# Input: nums1 = [1,0,0,2,3], nums2 = [0,3,0,4,0]   Output: 8
# Explanation: v1 = SparseVector(nums1) , v2 = SparseVector(nums2)
# v1.dotProduct(v2) = 1*0 + 0*3 + 0*0 + 2*4 + 3*0 = 8
# Input: nums1 = [0,1,0,0,0], nums2 = [0,0,0,0,2]   Output: 0
# Explanation: v1 = SparseVector(nums1) , v2 = SparseVector(nums2)
# v1.dotProduct(v2) = 0*0 + 1*0 + 0*0 + 0*0 + 0*2 = 0
# Input: nums1 = [0,1,0,0,2,0,0], nums2 = [1,0,0,0,3,0,4]   Output: 6
class SparseVector:
    def __init__(self, nums: List[int]):
        self.pairs = {}  # (x-coord, value)
        for i, n in enumerate(nums):
            if n != 0:
                self.pairs[i] = n

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        p = 0
        for x, v in vec.pairs.items():
            if x in self.pairs:
                p += self.pairs[x] * v
        return p

# https://leetcode.com/problems/valid-palindrome-iii/
# Given a string s and an integer k, return true if s is a k-palindrome.
# A string is k-palindrome if it can be transformed into a palindrome by removing at most k characters from it.
# Input: s = "abcdeca", k = 2
# Output: true
# Explanation: Remove 'b' and 'e' characters.
# Input: s = "abbababa", k = 1
# Output: true
# O(n*2) time and space
def isValidPalindrome(self, s: str, k: int) -> bool:
    n = len(s)
    memo = [[None] * n for _ in range(n)]
    def valid(i, j):
        if i == j:
            return 0
        if i == j - 1:
            return 1 if s[i] != s[j] else 0
        if memo[i][j] is not None:
            return memo[i][j]
        if s[i] == s[j]:
            memo[i][j] = valid(i + 1, j - 1)
        else:
            memo[i][j] = 1 + min(valid(i + 1, j), valid(i, j - 1))
        return memo[i][j]
    return valid(0, n - 1) <= k
# fill the same table as above, but do it bottom-up instead of top-down
def isValidPalindromeBottomUp(s: str, k: int) -> bool:
    n = len(s)
    memo = [[0] * n for _ in range(n)]
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            if s[i] == s[j]:
                memo[i][j] = memo[i + 1][j - 1] if i + 1 <= j - 1 else 0
            else:
                memo[i][j] = 1 + min(memo[i + 1][j], memo[i][j - 1])
    return memo[0][n - 1] <= k
# O(n^2) time and O(n) space
def isValidPalindromeIdeal(s: str, k: int) -> bool:
    n = len(s)
    memo = [0] * n
    for i in range(n - 2, -1, -1):
        prev = 0
        for j in range(i + 1, n):
            temp = memo[j]
            if s[i] == s[j]:
                memo[j] = prev
            else:
                memo[j] = 1 + min(memo[j], memo[j - 1])
            prev = temp
    return memo[n - 1] <= k
# no memoization: O(2^n) time - each function branches into 2 recursive calls and each call can be n deep at worst
def isValidPalindromeBrute(self, s: str, k: int) -> bool:
    def valid(lo, hi, removed):
        if removed > k:
            return float('inf') # not worth exploring, it's invalid
        if lo >= hi:
            return removed # we have met in the middle
        if s[lo] != s[hi]:
            # recursively try removing the char from left side and the char from right side
            return min(valid(lo + 1, hi, removed + 1), valid(lo, hi - 1, removed + 1))
        else:
            # these 2 chars can be kept in this palindrome, nothing to remove
            return valid(lo + 1, hi - 1, removed)
    minRemoved = valid(0, len(s) - 1, 0)
    return minRemoved <= k

# https://leetcode.com/problems/graph-valid-tree
# You have a graph of n nodes labeled from 0 to n - 1. You are given an integer n and a list of edges where edges[i] = [ai, bi] indicates that there is an undirected edge between nodes ai and bi in the graph.
# Return true if the edges of the given graph make up a valid tree, and false otherwise.
# O(n+e) time and space
def validTree(self, n: int, edges: List[List[int]]) -> bool:
    # build graph, traverse it, ensure no loops (route to a node previously visited) and we visit all nodes
    g = defaultdict(set)  # node -> set(connecting nodes)
    for e in edges:
        g[e[0]].add(e[1])
        g[e[1]].add(e[0])

    q = deque([0])
    visited = set()
    while q:
        cur = q.popleft()
        if cur in visited:  # a cycle exists
            return False
        visited.add(cur)

        for child in g[cur]:
            if child in visited:
                return False
            q.append(child)
            # Undirected edge back to parent doesn't count as a cycle so remove it before visiting. An alternative to
            # this would be using a parent map instead of visited set and skip visiting children that are parents of cur
            g[child].remove(cur)

    return n == len(visited)  # graph is conncected with no cycles (tree) if true
# graph must be connected and have exactly n-1 edges
# O(n) time and space where n=number of nodes
def validTree2(self, n: int, edges: List[List[int]]) -> bool:
    if len(edges) != n - 1: return False
    adj_list = [[] for _ in range(n)]
    for A, B in edges:
        adj_list[A].append(B)
        adj_list[B].append(A)

    seen = {0} # seen set to prevent our code from infinite looping if there *is* a cycle
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbour in adj_list[node]:
            if neighbour in seen:
                continue
            seen.add(neighbour)
            stack.append(neighbour)
    return len(seen) == n

# https://leetcode.com/problems/shortest-distance-from-all-buildings/

# https://leetcode.com/problems/shortest-word-distance-iii
# Given an array of strings wordsDict and two strings that already exist in the array word1 and word2, return the shortest distance between the occurrence of these two words in the list.
# Note that word1 and word2 may be the same. It is guaranteed that they represent two individual words in the list.
# O(n) time and O(1) space
def shortestWordDistance(self, wordsDict: list[str], word1: str, word2: str) -> int:
    shortest_distance = float('inf')
    prev_index = -1
    for i, word in enumerate(wordsDict):
        if word == word1 or word == word2:
            if prev_index != -1 and (wordsDict[prev_index] != word or word1 == word2):
                shortest_distance = min(shortest_distance, i - prev_index)
            prev_index = i
    return shortest_distance
# O(n*logn) time and O(n) space
def shortestWordDistanceFirst(self, wordsDict: List[str], word1: str, word2: str) -> int:
    w1s = []
    w2s = []
    for i, w in enumerate(wordsDict):
        if w == word1:
            w1s.append(i)
        if w == word2:
            w2s.append(i)

    result = float('inf')
    for w1 in w1s:
        closest = bisect_left(w2s, w1)  # index of first element >= w1
        cur = float('inf') if closest == len(w2s) else w2s[closest]  # cur is index in w2 closest above
        prev = float('-inf') if closest == 0 else w2s[closest - 1]  # prev is index in w2 closest below
        # must skip counting this distance if it's the same value
        if cur - w1 == 0:
            cur = float('inf')
        if w1 - prev == 0:
            prev = float('-inf')
        result = min(result, cur - w1, w1 - prev)
    return result

# https://leetcode.com/problems/maximum-size-subarray-sum-equals-k
# Given an integer array nums and an integer k, return the maximum length of a subarray
# that sums to k. If there is not one, return 0 instead.
# O(n) time and space
def maxSubArrayLen(self, nums: List[int], k: int) -> int:
    curSum = 0
    result = 0
    sums = {}  # sum -> first ending index (lowest i) of subarray nums[0:i] with the sum
    for i, n in enumerate(nums):
        curSum += n
        if curSum == k:
            result = i + 1

        target = curSum - k  # looking for a smaller subarray that doesn't start at 0 that could end at i and sum to k (curSum-prevSum = k)
        # get its end point and the difference between big suffix sum and little suffix sum is the middle subarray suffix sum that we want
        if target in sums:
            result = max(result, i - sums[target])

        if curSum not in sums:
            sums[curSum] = i
    return result

# https://leetcode.com/problems/walls-and-gates
# You are given an m x n grid rooms initialized with these three possible values.
# -1 A wall or an obstacle.
# 0 A gate.
# INF Infinity means an empty room. We use the value 231 - 1 = 2147483647 to represent INF as you may assume that the distance to a gate is less than 2147483647.
# Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.
EMPTY = 2**31 - 1  # Integer.MAX_VALUE equivalent
GATE = 0
DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
def wallsAndGatesBest(rooms: List[List[int]]) -> None:
    if not rooms:
        return
    m, n = len(rooms), len(rooms[0])
    queue = deque()
    for row in range(m):
        for col in range(n):
            if rooms[row][col] == GATE:
                queue.append((row, col))
    while queue:
        row, col = queue.popleft()
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            if 0 <= r < m and 0 <= c < n and rooms[r][c] == EMPTY: # only explore the previously unvisited nodes since BFS will find shortest path
                rooms[r][c] = rooms[row][col] + 1
                queue.append((r, c))
# single BFS from many sources - O(nm) time and space
def wallsAndGatesOK(self, rooms: List[List[int]]) -> None:
    q = deque()
    for y in range(len(rooms)):
        for x in range(len(rooms[0])):
            if rooms[y][x] == 0:
                q.append((x, y, 0))
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    v = set()
    while q:
        level = len(q)
        for _ in range(level):
            cx, cy, dist = q.popleft()
            v.add((cx, cy))
            rooms[cy][cx] = min(rooms[cy][cx], dist)
            for d in dirs:
                nx, ny = cx + d[0], cy + d[1]
                if nx >= 0 and nx < len(rooms[0]) and ny >= 0 and ny < len(rooms) and rooms[ny][nx] != -1 and (
                nx, ny) not in v:
                    q.append((nx, ny, dist + 1))
        dist += 1
# multi-source multiple BFSs - O(n^2*m^2) time and O(nm) space
def wallsAndGatesBad(self, rooms: List[List[int]]) -> None:
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for y in range(len(rooms)):
        for x in range(len(rooms[0])):
            if rooms[y][x] == 0:
                v = set()
                dist = 0
                q = deque([(x, y)])
                while q:
                    level = len(q)
                    for _ in range(level):
                        cx, cy = q.popleft()
                        v.add((cx, cy))
                        rooms[cy][cx] = min(rooms[cy][cx], dist)
                        for d in dirs:
                            nx, ny = cx + d[0], cy + d[1]
                            if nx >= 0 and nx < len(rooms[0]) and ny >= 0 and ny < len(rooms) and rooms[ny][
                                nx] != -1 and (nx, ny) not in v and rooms[ny][nx] > dist + 1:
                                q.append((nx, ny))
                    dist += 1

# https://leetcode.com/problems/synonymous-sentences
# You are given a list of equivalent string pairs synonyms where synonyms[i] = [si, ti] indicates that si and ti are equivalent strings. You are also given a sentence text.
# Return all possible synonymous sentences sorted lexicographically.
def generateSentences(self, synonyms: List[List[str]], text: str) -> List[str]:
    graph = defaultdict(dict)
    q = deque()
    ans = set()
    q.append(text)
    for k, v in synonyms:
        graph[k][v] = 1
        graph[v][k] = 1
    while q:
        curT = q.popleft()
        ans.add(curT)
        words = curT.split()
        for i, w in enumerate(words):
            if w in graph.keys():
                for newW in graph[w]:
                    newsent = ' '.join(words[:i] + [newW] + words[i + 1:])
                    if newsent not in ans:
                        q.append(newsent)
    return sorted(list(ans))

# https://leetcode.com/problems/shortest-distance-from-all-buildings
# O(n^2 * m^2) time and O(n*m) space
def shortestDistance(self, grid: List[List[int]]) -> int:
    rows = len(grid)
    cols = len(grid[0])
    dirs = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    total_sum = [[0] * cols for _ in range(rows)] # dist to travel from all houses to reach each land spot, return min of this

    def bfs(row, col, curr_count):
        min_distance = math.inf
        queue = deque()
        queue.append([row, col, 0])
        while queue:
            curr_row, curr_col, dist = queue.popleft()
            for dx, dy in dirs:
                next_row = curr_row + dx
                next_col = curr_col + dy
                if 0 <= next_row < rows and 0 <= next_col < cols and grid[next_row][next_col] == -curr_count:
                    # it's reachable empty land
                    total_sum[next_row][next_col] += dist + 1
                    min_distance = min(min_distance, total_sum[next_row][next_col])
                    grid[next_row][next_col] -= 1 # use original grid to track reachable & visited land
                    queue.append([next_row, next_col, dist + 1])
        return min_distance

    count = 0 # number of bfses that have been done
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 1: # expand out from each house
                min_distance = bfs(row, col, count)
                count += 1
                if min_distance == math.inf: # couldn't reach any available land from a house
                    return -1
    return min_distance
def shortestDistanceBad(self, grid: List[List[int]]) -> int:
    # do a simultaneous bfs from all buildings keeping track of which building the bfs started from, when the first point to be visited by all bfs paths is hit, return it
    # problem with doing simultaneous bfs is a point may end up first in q that isn't the shortest dist (but is close) leading to a few off-by-one distances
    yMax = len(grid)
    xMax = len(grid[0])
    buildings = deque()
    nextBuildingId = 0
    for y in range(yMax):
        for x in range(xMax):
            if grid[y][x] == 1:
                buildings.append((x, y, 0, nextBuildingId))
                nextBuildingId += 1

    # map:  (x,y) -> map(building ids that have hit this spot -> distance travelled)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    spots = {}
    while buildings:
        cx, cy, d, cb = buildings.popleft()
        coord = (cx, cy)
        visits = spots.get(coord, {})  # always start on a 1 that we wont revisit so no need to save it
        # print('visiting', coord, 'from building', cb, 'after travelling', d, 'total visits at this spot', len(visits))
        if len(visits) == nextBuildingId:
            return sum(visits.values())
        for (dx, dy) in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < xMax and 0 <= ny < yMax and grid[ny][nx] == 0:
                nvs = spots.get((nx, ny), {})
                if cb not in nvs:
                    nvs[cb] = d + 1
                    spots[(nx, ny)] = nvs  # write the inner map in case it's first time and it's not set before
                    buildings.append((nx, ny, d + 1, cb))
    return -1

# https://leetcode.com/problems/missing-ranges
# You are given an inclusive range [lower, upper] and a sorted unique integer array nums, where all elements are within the inclusive range.
# A number x is considered missing if x is in the range [lower, upper] and x is not in nums.
# Return the shortest sorted list of ranges that exactly covers all the missing numbers. That is, no element of nums is included in any of the ranges, and each missing number is covered by one of the ranges.
def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[List[int]]:
    if not nums:
        return [[lower, upper]]
    missing = []
    if lower < nums[0]:
        missing.append([lower, nums[0] - 1])

    for i in range(1, len(nums)):
        if nums[i - 1] != nums[i] and nums[i - 1] != nums[i] - 1:
            missing.append([nums[i - 1] + 1, nums[i] - 1])

    if nums[-1] < upper:
        missing.append([nums[-1] + 1, upper])
    return missing

# https://leetcode.com/problems/group-shifted-strings
# Perform the following shift operations on a string:
# Right shift: Replace every letter with the successive letter of the English alphabet, where 'z' is replaced by 'a'. For example, "abc" can be right-shifted to "bcd" or "xyz" can be right-shifted to "yza".
# Left shift: Replace every letter with the preceding letter of the English alphabet, where 'a' is replaced by 'z'. For example, "bcd" can be left-shifted to "abc" or "yza" can be left-shifted to "xyz".
# We can keep shifting the string in both directions to form an endless shifting sequence.
# For example, shift "abc" to form the sequence: ... <-> "abc" <-> "bcd" <-> ... <-> "xyz" <-> "yza" <-> .... <-> "zab" <-> "abc" <-> ...
# You are given an array of strings strings, group together all strings[i] that belong to the same shifting sequence. You may return the answer in any order.
# O(N*K) time and space
def groupStrings(self, strings: List[str]) -> List[List[str]]:
    def shiftedVal(s):
        diff = ord(s[0]) - ord('a')
        new = []
        for c in s:
            char = ord(c) - diff
            if char < ord('a'):
                char += 26
            new.append(chr(char))
        return ''.join(new)

    d = defaultdict(list)
    for s in strings:
        key = shiftedVal(s)
        d[key].append(s)
    return [v for v in d.values()]

# https://leetcode.com/problems/buildings-with-an-ocean-view
# There are n buildings in a line. You are given an integer array heights of size n that represents the heights of the buildings in the line.
# The ocean is to the right of the buildings. A building has an ocean view if the building can see the ocean without obstructions. Formally, a building has an ocean view if all the buildings to its right have a smaller height.
# Return a list of indices (0-indexed) of buildings that have an ocean view, sorted in increasing order.
def findBuildings(self, heights: List[int]) -> List[int]:
    result = []
    curMax = float('-inf')
    for i in range(len(heights) - 1, -1, -1):
        if heights[i] > curMax:
            result.append(i)
        curMax = max(curMax, heights[i])
    result.reverse()
    return result

# https://leetcode.com/problems/construct-binary-tree-from-string
# You need to construct a binary tree from a string consisting of parenthesis and integers.
# The whole input represents a binary tree. It contains an integer followed by zero, one or two pairs of parenthesis. The integer represents the root's value and a pair of parenthesis contains a child binary tree with the same structure.
# You always start to construct the left child node of the parent first if it exists.
def str2tree(self, s: str) -> Optional[TreeNode]:
    if not s:
        return None
    p = s.find('(')
    if p == -1: # a leaf node
        return TreeNode(int(s))
    node = TreeNode(int(s[:p]))

    opens = 1
    mid = -1 # find the dividing index between left and right
    for i in range(p + 1, len(s)):
        if s[i] == '(':
            opens += 1
        elif s[i] == ')':
            opens -= 1
        if opens == 0:
            mid = i
            break
    if mid == -1: # everything is the left child
        node.left = self.str2tree(s[p + 1 : -1])  # start after the first ( and remove the final )
    else: # there is a left and right
        node.left = self.str2tree(s[p + 1 : mid])
        node.right = self.str2tree(s[mid + 2 : -1])  # start after end of ')(' sequence and remove the final )
    return node

# https://leetcode.com/problems/cutting-ribbons
# You are given an integer array ribbons, where ribbons[i] represents the length of the ith ribbon, and an integer k. You may cut any of the ribbons into any number of segments of positive integer lengths, or perform no cuts at all.
# For example, if you have a ribbon of length 4, you can:
# Keep the ribbon of length 4,
# Cut it into one ribbon of length 3 and one ribbon of length 1,
# Cut it into two ribbons of length 2,
# Cut it into one ribbon of length 2 and two ribbons of length 1, or
# Cut it into four ribbons of length 1.
# Your task is to determine the maximum length of ribbon, x, that allows you to cut at least k ribbons, each of length x. You can discard any leftover ribbon from the cuts. If it is impossible to cut k ribbons of the same length, return 0.


# O(n*logm) time and O(1) space; m = max in ribbons and n = len(ribbons)
def maxLength(self, ribbons: List[int], k: int) -> int:
    # need k ribbons after cutting (unlimited cuts), what is largest ribbon size x?
    # if k == 1, return max(ribbons)
    # if k == len(ribbons), return min(ribbons) <- not necessarily, 2nd largest could be more than double the smallest
    # if 1 < k < len(ribbons), return the kth largest
    # if k > sum(ribbons), return 0 (can't even make ribbons of length 1)
    # if k == len(ribbons)+1, return min(min(ribbons), max(ribbons) // 2) <- not necessarily, 2nd largest could be more than double the smallest

    # returns true if ribbons can be cut into at least k pieces of len n
    def canCut(n):
        pieces = 0  # count of pieces of len n
        for i in range(len(ribbons)):
            pieces += ribbons[i] // n
            if pieces >= k:
                return True
        return False

    # optimized:
    lo = 0
    hi = max(ribbons)  # no piece can be longer than the length of the longest ribbon
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if mid == 0:
            mid = 1  # can't divide by zero so try 1
        if canCut(mid):
            lo = mid + 1
        else:
            if mid == 1:
                return 0  # size 1 didn't work so ans is 0
            hi = mid - 1
    return hi

    # non-optimized:
    # lo = 1
    # hi = sum(ribbons)
    # if k > hi: return 0
    # while lo <= hi:
    #     mid = lo + (hi - lo) // 2
    #     if canCut(mid):
    #         lo = mid + 1
    #     else:
    #         hi = mid - 1
    # return hi

# https://leetcode.com/problems/product-of-two-run-length-encoded-arrays
# This is an interactive problem.
# There is a robot in a hidden grid, and you are trying to get it from its starting cell to the target cell in this grid. The grid is of size m x n, and each cell in the grid is either empty or blocked. It is guaranteed that the starting cell and the target cell are different, and neither of them is blocked.
# You want to find the minimum distance to the target cell. However, you do not know the grid's dimensions, the starting cell, nor the target cell. You are only allowed to ask queries to the GridMaster object.
# Thr GridMaster class has the following functions:
# boolean canMove(char direction) Returns true if the robot can move in that direction. Otherwise, it returns false.
# void move(char direction) Moves the robot in that direction. If this move would move the robot to a blocked cell or off the grid, the move will be ignored, and the robot will remain in the same position.
# boolean isTarget() Returns true if the robot is currently on the target cell. Otherwise, it returns false.
# Note that direction in the above functions should be a character from {'U','D','L','R'}, representing the directions up, down, left, and right, respectively.
# Return the minimum distance between the robot's initial starting cell and the target cell. If there is no valid path between the cells, return -1.
# Custom testing:
# The test input is read as a 2D matrix grid of size m x n where:
# grid[i][j] == -1 indicates that the robot is in cell (i, j) (the starting cell).
# grid[i][j] == 0 indicates that the cell (i, j) is blocked.
# grid[i][j] == 1 indicates that the cell (i, j) is empty.
# grid[i][j] == 2 indicates that the cell (i, j) is the target cell.
# There is exactly one -1 and 2 in grid. Remember that you will not have this information in your code.
class GridMaster(object):
   def canMove(self, direction: str) -> bool:
        return False
   def move(self, direction: str) -> None:
        return None
   def isTarget(self) -> bool:
        return False
def findShortestPath(self, master: 'GridMaster') -> int:
    # first use dfs to find all possible reachable positions
    dirs = {"U":(-1, 0), "D":(1, 0), "L":(0, -1), "R":(0, 1)}
    anti = {"U":"D", "D":"U", "L":"R", "R":"L"}

    isValid = {}
    isValid[(0, 0)] = master.isTarget()

    def dfs(r, c):

        for d in dirs:
            dr, dc = dirs[d]
            nr, nc = r + dr, c + dc
            if (nr, nc) not in isValid and master.canMove(d):
                # move forward
                master.move(d)
                isValid[(nr, nc)] = master.isTarget()
                dfs(nr, nc)
                # move back
                master.move(anti[d])
    dfs(0, 0)

    # now use bfs to find the minimum distance
    qu = collections.deque([(0,0, 0)]) # (r, c, step)
    seen = set()
    while qu:
        r, c, step = qu.popleft()
        if isValid[(r,c)] == True:
            return step

        for nr, nc in [[r+1, c], [r-1, c], [r, c-1], [r, c+1]]:
            if (nr, nc) in isValid and (nr, nc) not in seen:
                seen.add((nr, nc))
                qu.append((nr, nc, step+1))
    return -1

# https://leetcode.com/problems/product-of-two-run-length-encoded-arrays
# Run-length encoding is a compression algorithm that allows for an integer array nums with many segments of consecutive repeated numbers to be represented by a (generally smaller) 2D array encoded. Each encoded[i] = [vali, freqi] describes the ith segment of repeated numbers in nums where vali is the value that is repeated freqi times.
# For example, nums = [1,1,1,2,2,2,2,2] is represented by the run-length encoded array encoded = [[1,3],[2,5]]. Another way to read this is "three 1's followed by five 2's".
# The product of two run-length encoded arrays encoded1 and encoded2 can be calculated using the following steps:
# Expand both encoded1 and encoded2 into the full arrays nums1 and nums2 respectively.
# Create a new array prodNums of length nums1.length and set prodNums[i] = nums1[i] * nums2[i].
# Compress prodNums into a run-length encoded array and return it.
# You are given two run-length encoded arrays encoded1 and encoded2 representing full arrays nums1 and nums2 respectively. Both nums1 and nums2 have the same length. Each encoded1[i] = [vali, freqi] describes the ith segment of nums1, and each encoded2[j] = [valj, freqj] describes the jth segment of nums2.
# Return the product of encoded1 and encoded2.
# Note: Compression should be done such that the run-length encoded array has the minimum possible length.
# Example 1:
# Input: encoded1 = [[1,3],[2,3]], encoded2 = [[6,3],[3,3]]
# Output: [[6,6]]
# Explanation: encoded1 expands to [1,1,1,2,2,2] and encoded2 expands to [6,6,6,3,3,3].
# prodNums = [6,6,6,6,6,6], which is compressed into the run-length encoded array [[6,6]].
# Example 2:
# Input: encoded1 = [[1,3],[2,1],[3,2]], encoded2 = [[2,3],[3,3]]
# Output: [[2,3],[6,1],[9,2]]
# Explanation: encoded1 expands to [1,1,1,2,3,3] and encoded2 expands to [2,2,2,3,3,3].
# prodNums = [2,2,2,6,9,9], which is compressed into the run-length encoded array [[2,3],[6,1],[9,2]].
# Constraints:
# 1 <= encoded1.length, encoded2.length <= 105
# encoded1[i].length == 2
# encoded2[j].length == 2
# 1 <= vali, freqi <= 104 for each encoded1[i].
# 1 <= valj, freqj <= 104 for each encoded2[j].
# The full arrays that encoded1 and encoded2 represent are the same length.
def findRLEArray(self, encoded1: List[List[int]], encoded2: List[List[int]]) -> List[List[int]]:
    i = j = f1 = f2 = v1 = v2 = 0                # Declare variables
    m, n, ans = len(encoded1), len(encoded2), []
    while i < m or j < n:                        # Starting two pointers while loop
        if not f1 and i < m:                     # If `f1 == 0`, assign new value and frequency
            v1, f1 = encoded1[i]
        if not f2 and j < n:                     # If `f2 == 0`, assign new value and frequency
            v2, f2 = encoded2[j]
        cur_min, product = min(f1, f2), v1 * v2  # Calculate smaller frequency and product
        if ans and ans[-1][0] == product:        # If current product is the same as previous one, update previous frequency
            ans[-1][1] += cur_min
        else:                                    # Other situation, append new pairs
            ans.append([product, cur_min])
        f1 -= cur_min                            # Deduct frequency by smaller frequency (used in current round)
        f2 -= cur_min
        i += not f1                              # When frequency is zero, increment pointer by 1
        j += not f2
    return ans
def findRLEArray(self, encoded1: List[List[int]], encoded2: List[List[int]]) -> List[List[int]]:
    product_encoded = []
    e1_index = 0
    e2_index = 0
    while e1_index < len(encoded1) and e2_index < len(encoded2):
        e1_val, e1_freq = encoded1[e1_index]
        e2_val, e2_freq = encoded2[e2_index]

        product_val = e1_val * e2_val
        product_freq = min(e1_freq, e2_freq)

        encoded1[e1_index][1] -= product_freq
        encoded2[e2_index][1] -= product_freq

        if encoded1[e1_index][1] == 0:
            e1_index += 1

        if encoded2[e2_index][1] == 0:
            e2_index += 1

        if not product_encoded or product_encoded[-1][0] != product_val:
            product_encoded.append([product_val, product_freq])
        else:
            product_encoded[-1][1] += product_freq
    return product_encoded

