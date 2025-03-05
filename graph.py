import collections
from collections import deque, defaultdict
from typing import List
from typing import Optional

'''
Graph representations:
    * Adjacency list - usually best; typically a map of node -> [children of node] 
    * Adjacency matrix - 2d array (row index as source node and col as destination with True/False/Weight as the
                            value); use if # edges is much > than # nodes 
    * Linked nodes - use Node class and pointers to other Nodes
'''

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

# https://leetcode.com/problems/course-schedule
def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    graph = {} # course -> [courses that must be taken afterward]
    indegree = {} # course -> number of prerequisites     note: no visited set required in traversal since we have this

    for c, p in prerequisites:
        indegree[c] = indegree.get(c, 0) + 1
        if p in graph:
            graph[p].append(c)
        else:
            graph[p] = [c]

    wl = deque([i for i in range(numCourses) if i not in indegree])

    count = 0
    while wl:
        course = wl.popleft()
        count += 1
        if course in graph:
            for nextCourse in graph[course]:
                if indegree[nextCourse] == 1:
                    wl.append(nextCourse)
                    del indegree[nextCourse]
                else:
                    indegree[nextCourse] -= 1
    return count == numCourses

# https://leetcode.com/problems/course-schedule-ii
def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    result = []
    graph = {} # course -> [courses that must be taken afterward]
    indegree = {} # course -> number of courses that current must be taken before

    for c, p in prerequisites:
        indegree[c] = indegree.get(c, 0) + 1
        if p in graph:
            graph[p].append(c)
        else:
            graph[p] = [c]

    wl = deque([i for i in range(numCourses) if i not in indegree])

    while wl:
        course = wl.popleft()
        result.append(course)
        if course in graph:
            for nextCourse in graph[course]:
                if indegree[nextCourse] == 1:
                    wl.append(nextCourse)
                    del indegree[nextCourse]
                else:
                    indegree[nextCourse] -= 1
    return result if numCourses == len(result) else []

# https://leetcode.com/problems/clone-graph
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
    if not node:
        return None
    d = {}
    wl = deque()
    wl.append(node)
    # first map old node -> new node
    while wl:
        n = wl.popleft()
        d[n] = Node(n.val, [])
        for neigh in n.neighbors:
            if neigh not in d:
                wl.append(neigh)
    # fill in new neighbors from the map
    wl.append(node)
    visited = set()
    while wl:
        n = wl.popleft()
        if n in visited: # avoid revisiting nodes as prevention before adding to wl may not catch all
            continue
        visited.add(n)
        for neigh in n.neighbors:
            d[n].neighbors.append(d[neigh])
            if neigh not in visited:
                wl.append(neigh)
    return d[node]
def cloneGraphPretty(self, node: 'Node') -> 'Node':
    if not node: return node
    q, clones = deque([node]), {node.val: Node(node.val, [])}
    while q:
        cur = q.popleft()
        cur_clone = clones[cur.val]

        for ngbr in cur.neighbors:
            if ngbr.val not in clones:
                clones[ngbr.val] = Node(ngbr.val, [])
                q.append(ngbr)

            cur_clone.neighbors.append(clones[ngbr.val])
    return clones[node.val]

# https://leetcode.com/problems/copy-list-with-random-pointer
def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
    if not head:
        return None
    oldToNew = {}

    cur = head
    while cur:
        oldToNew[cur] = Node(cur.val)
        cur = cur.next

    cur = head
    new = oldToNew[head]
    while cur:
        if cur.random:
            new.random = oldToNew[cur.random]
        cur = cur.next
        if cur:
            new.next = oldToNew[cur]
            new = new.next
    return oldToNew[head]
# https://leetcode.com/problems/copy-list-with-random-pointer/solutions/4003262/9792-hash-table-linked-list-by-vanamsen-boof/
def copyRandomListInterleave(self, head: 'Optional[Node]') -> 'Optional[Node]':
    if not head:
        return None

    curr = head
    while curr:
        new_node = Node(curr.val, curr.next)
        curr.next = new_node
        curr = new_node.next

    curr = head
    while curr:
        if curr.random:
            curr.next.random = curr.random.next
        curr = curr.next.next

    old_head = head
    new_head = head.next
    curr_old = old_head
    curr_new = new_head

    while curr_old:
        curr_old.next = curr_old.next.next
        curr_new.next = curr_new.next.next if curr_new.next else None
        curr_old = curr_old.next
        curr_new = curr_new.next

    return new_head

# https://leetcode.com/problems/open-the-lock
def openLock(self, deadends: List[str], target: str) -> int:
    deadends = set(deadends)
    if '0000' in deadends:
        return -1
    if '0000' == target:
        return 0

    # bfs to find shortest distance
    visited = set('0000')
    wl = deque([([0, 0, 0, 0], 0)])  # (combination, distance)
    while wl:
        comb, dist = wl.popleft()

        # make moves, any of the 4 numbers could go up or down, explore them if they are valid
        for i in range(4):
            nextComb = list(comb)
            nextComb[i] = (nextComb[i] + 1) % 10
            nextCombStr = ''.join([str(s) for s in nextComb])
            if nextCombStr not in deadends and nextCombStr not in visited:
                if nextCombStr == target:
                    return dist + 1
                wl.append((nextComb, dist + 1))
                visited.add(nextCombStr)

            nextComb = list(comb)
            nextComb[i] = (nextComb[i] - 1) % 10
            nextCombStr = ''.join([str(s) for s in nextComb])
            if nextCombStr not in deadends and nextCombStr not in visited:
                if nextCombStr == target:
                    return dist + 1
                wl.append((nextComb, dist + 1))
                visited.add(nextCombStr)
    return -1
# O(4(d+10^4)) time and space
def openLockPretty(self, deadends: List[str], target: str) -> int:
    deadends = set(deadends)
    if "0000" in deadends:
        return -1
    queue = deque([("0000", 0)])
    visited = set("0000")
    while queue:
        state, moves = queue.popleft()
        if state == target:
            return moves
        for i in range(4):
            for delta in (-1, 1):
                new_digit = (int(state[i]) + delta) % 10
                new_state = state[:i] + str(new_digit) + state[i + 1:]

                if new_state not in visited and new_state not in deadends:
                    visited.add(new_state)
                    queue.append((new_state, moves + 1))
    return -1

# https://leetcode.com/problems/number-of-provinces
# O(n^2) time and O(n) space
def findCircleNum(self, isConnected: List[List[int]]) -> int:
    n = len(isConnected)
    visited = set()
    result = 0
    for i in range(n):
        if i in visited:
            continue
        result += 1 # found a new province
        # do a dfs from i exploring the province and adding its nodes to visited
        stack = [i]
        visited.add(i)
        while stack:
            cur = stack.pop()
            # check for connected children
            for j in range(n):
                if isConnected[cur][j] and j not in visited:
                    visited.add(j)
                    stack.append(j)
    return result

# https://leetcode.com/problems/minimum-time-to-collect-all-apples-in-a-tree
# O(n) time and space
def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
    def dfs(node, parent, has_apple):
        if node not in adj:
            return 0
        total_time = 0
        for child in adj[node]:
            if child == parent:
                continue
            child_time = dfs(child, node, has_apple)
            # If the subtree has apples, add the time cost
            if child_time > 0 or has_apple[child]:
                total_time += child_time + 2
        return total_time

    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return dfs(0, -1, hasApple)
def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
    graph = defaultdict(list)
    for e in edges:
        graph[e[0]].append(e[1])
        graph[e[1]].append(e[0])

    allEdges = set()
    visited = [False] * len(hasApple)

    def findApple(node):
        visited[node] = True
        # look for all apples, return True if a node is an Apple
        saveEdgeAbove = False
        if hasApple[node]:
            saveEdgeAbove = True
        for child in graph[node]:
            if not visited[child]:
                foundApple = findApple(child)
                if foundApple:
                    # if child is an apple, we need to save the edges going down to it and above current node to root
                    allEdges.add((child, node))
                    saveEdgeAbove = True
        return saveEdgeAbove
    findApple(0)
    return 2 * len(allEdges)
def minTimeSlow(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
    graph = defaultdict(list)
    for e in edges:
        graph[e[0]].append(e[1])
        graph[e[1]].append(e[0])

    allEdges = set()
    visited = [False] * len(hasApple)
    visited[0] = True
    stack = [(0, set())]  # node, edges from root
    while stack:
        cur, edges = stack.pop()
        if hasApple[cur]:
            allEdges.update(edges)

        for child in graph[cur]:
            if not visited[child]:
                visited[child] = True
                childEdges = set(edges)
                childEdges.add((cur, child))
                stack.append((child, childEdges))
    return 2 * len(allEdges)

# https://leetcode.com/problems/divide-nodes-into-the-maximum-number-of-groups
# Main function to calculate the maximum number of magnificent sets
# Time complexity: O(n×(n+m)) Space complexity: O(n+m)
def magnificentSets(self, n, edges):
    # Create adjacency list for the graph
    adj_list = [[] for _ in range(n)]
    for edge in edges:
        # Transition to 0-index
        adj_list[edge[0] - 1].append(edge[1] - 1)
        adj_list[edge[1] - 1].append(edge[0] - 1)

    # Initialize color array to -1
    colors = [-1] * n

    # Check if the graph is bipartite
    for node in range(n):
        if colors[node] != -1:
            continue
        # Start coloring from uncolored nodes
        colors[node] = 0
        if not self._is_bipartite(adj_list, node, colors):
            return -1

    # Calculate the longest shortest path for each node
    distances = [
        self._get_longest_shortest_path(adj_list, node, n)
        for node in range(n)
    ]

    # Calculate the total maximum number of groups across all components
    max_number_of_groups = 0
    visited = [False] * n
    for node in range(n):
        if visited[node]:
            continue
        # Add the number of groups for this component to the total
        max_number_of_groups += self._get_number_of_groups_for_component(
            adj_list, node, distances, visited
        )

    return max_number_of_groups
# Checks if the graph is bipartite starting from the given node
def _is_bipartite(self, adj_list, node, colors):
    for neighbor in adj_list[node]:
        # If a neighbor has the same color as the current node, the graph is not bipartite
        if colors[neighbor] == colors[node]:
            return False
        # If the neighbor is already colored, skip it
        if colors[neighbor] != -1:
            continue
        # Assign the opposite color to the neighbor
        colors[neighbor] = (colors[node] + 1) % 2
        # Recursively check bipartiteness for the neighbor; return false if it fails
        if not self._is_bipartite(adj_list, neighbor, colors):
            return False
    # If all neighbors are properly colored, return true
    return True
# Computes the longest shortest path (height) in the graph starting from the source node
def _get_longest_shortest_path(self, adj_list, src_node, n):
    # Initialize a queue for BFS and a visited array
    nodes_queue = deque([src_node])
    visited = [False] * n
    visited[src_node] = True
    distance = 0

    # Perform BFS layer by layer
    while nodes_queue:
        # Process all nodes in the current layer
        for _ in range(len(nodes_queue)):
            current_node = nodes_queue.popleft()
            # Visit all unvisited neighbors of the current node
            for neighbor in adj_list[current_node]:
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                nodes_queue.append(neighbor)
        # Increment the distance for each layer
        distance += 1

    # Return the total distance (longest shortest path)
    return distance
# Calculates the maximum number of groups for a connected component
def _get_number_of_groups_for_component(
    self, adj_list, node, distances, visited
):
    # Start with the distance of the current node as the maximum
    max_number_of_groups = distances[node]
    visited[node] = True

    # Recursively calculate the maximum for all unvisited neighbors
    for neighbor in adj_list[node]:
        if visited[neighbor]:
            continue
        max_number_of_groups = max(
            max_number_of_groups,
            self._get_number_of_groups_for_component(
                adj_list, neighbor, distances, visited
            ),
        )
    return max_number_of_groups
# approach 2 union find: Time complexity: O(n×(n+m)) Space complexity: O(n+m)
class Solution:
    # Main function to calculate the maximum number of groups for the entire graph
    def magnificentSets(self, n, edges):
        adj_list = [[] for _ in range(n)]
        parent = [-1] * n
        depth = [0] * n

        # Build the adjacency list and apply Union-Find for each edge
        for edge in edges:
            adj_list[edge[0] - 1].append(edge[1] - 1)
            adj_list[edge[1] - 1].append(edge[0] - 1)
            self._union(edge[0] - 1, edge[1] - 1, parent, depth)

        num_of_groups_for_component = {}

        # For each node, calculate the maximum number of groups for its component
        for node in range(n):
            number_of_groups = self._get_number_of_groups(adj_list, node, n)
            if number_of_groups == -1:
                return -1  # If invalid split, return -1
            root_node = self._find(node, parent)
            num_of_groups_for_component[root_node] = max(
                num_of_groups_for_component.get(root_node, 0), number_of_groups
            )

        # Calculate the total number of groups across all components
        total_number_of_groups = sum(num_of_groups_for_component.values())
        return total_number_of_groups

    # Function to calculate the number of groups for a given component starting from srcNode
    def _get_number_of_groups(self, adj_list, src_node, n):
        nodes_queue = deque()
        layer_seen = [-1] * n
        nodes_queue.append(src_node)
        layer_seen[src_node] = 0
        deepest_layer = 0

        # Perform BFS to calculate the number of layers (groups)
        while nodes_queue:
            num_of_nodes_in_layer = len(nodes_queue)
            for _ in range(num_of_nodes_in_layer):
                current_node = nodes_queue.popleft()
                for neighbor in adj_list[current_node]:
                    # If neighbor hasn't been visited, assign it to the next layer
                    if layer_seen[neighbor] == -1:
                        layer_seen[neighbor] = deepest_layer + 1
                        nodes_queue.append(neighbor)
                    else:
                        # If the neighbor is already in the same layer, return -1 (invalid partition)
                        if layer_seen[neighbor] == deepest_layer:
                            return -1
            deepest_layer += 1
        return deepest_layer

    # Find the root of the given node in the Union-Find structure
    def _find(self, node, parent):
        while parent[node] != -1:
            node = parent[node]
        return node

    # Union operation to merge two sets
    def _union(self, node1, node2, parent, depth):
        node1 = self._find(node1, parent)
        node2 = self._find(node2, parent)

        # If both nodes already belong to the same set, no action needed
        if node1 == node2:
            return

        # Union by rank (depth) to keep the tree balanced
        if depth[node1] < depth[node2]:
            node1, node2 = node2, node1
        parent[node2] = node1

        # If the depths are equal, increment the depth of the new root
        if depth[node1] == depth[node2]:
            depth[node1] += 1

# https://leetcode.com/problems/reconstruct-itinerary

# https://leetcode.com/problems/minimize-malware-spread
# O(n^2) time and O(n) space
def minMalwareSpread(self, graph, initial):
    # 1. Color each component.
    # colors[node] = the color of this node.

    N = len(graph)
    colors = {}
    c = 0

    def dfs(node, color):
        colors[node] = color
        for nei, adj in enumerate(graph[node]):
            if adj and nei not in colors:
                dfs(nei, color)

    for node in range(N):
        if node not in colors:
            dfs(node, c)
            c += 1

    # 2. Size of each color.
    # size[color] = number of occurrences of this color.
    size = collections.Counter(colors.values())

    # 3. Find unique colors.
    color_count = collections.Counter()
    for node in initial:
        color_count[colors[node]] += 1

    # 4. Answer
    ans = float('inf')
    for x in initial:
        c = colors[x]
        if color_count[c] == 1:
            if ans == float('inf'):
                ans = x
            elif size[c] > size[colors[ans]]:
                ans = x
            elif size[c] == size[colors[ans]] and x < ans:
                ans = x

    return ans if ans < float('inf') else min(initial)
class DSU:
    def __init__(self, N):
        self.p = range(N)
        self.sz = [1] * N

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        self.p[xr] = yr
        self.sz[yr] += self.sz[xr]

    def size(self, x):
        return self.sz[self.find(x)]
# O(n^2) time and O(n) space
def minMalwareSpreadUnionFind(self, graph, initial):
    dsu = DSU(len(graph))

    for j, row in enumerate(graph):
        for i in range(j):
            if row[i]:
                dsu.union(i, j)

    count = collections.Counter(dsu.find(u) for u in initial)
    ans = (-1, min(initial))
    for node in initial:
        root = dsu.find(node)
        if count[root] == 1:  # unique color
            if dsu.size(root) > ans[0]:
                ans = dsu.size(root), node
            elif dsu.size(root) == ans[0] and node < ans[1]:
                ans = dsu.size(root), node

    return ans[1]