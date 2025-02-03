from collections import deque
from typing import List
from typing import Optional


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
    indegree = {} # course -> number of courses that current must be taken before

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
