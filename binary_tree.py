from collections import deque, defaultdict
from typing import Optional, List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    def __str__(self):
        if self.val:
            return f',{self.val}' + str(self.left) + ',' + str(self.right)
        return ''

# https://leetcode.com/problems/reverse-odd-levels-of-binary-tree
# O(n) time and space
def reverseOddLevelsBFS(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
        return root
    wl = deque()
    wl.append(root)
    level = -1
    while len(wl) > 0:
        level += 1
        size = len(wl)
        if level % 2 == 1:
            # reverse values in the level
            for i in range(0, size // 2):
                temp = wl[i].val
                wl[i].val = wl[size - 1 - i].val
                wl[size - 1 - i].val = temp
        # traverse a level
        for i in range(size):
            node = wl.popleft()
            if node.left:
                wl.append(node.left)
            if node.right:
                wl.append(node.right)
    return root
# DFS O(n) time and O(logn) space
def reverseOddLevels(self, root) -> TreeNode:
    self.__traverse_DFS(root.left, root.right, 0)
    return root
def __traverse_DFS(self, left_child, right_child, level):
    if left_child is None or right_child is None:
        return
    # If the current level is odd, swap the values of the children.
    if level % 2 == 0:
        temp = left_child.val
        left_child.val = right_child.val
        right_child.val = temp
    self.__traverse_DFS(left_child.left, right_child.right, level + 1)
    self.__traverse_DFS(left_child.right, right_child.left, level + 1)

# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree
# O(n) time and space
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    # find p and q
    # if p and q are found and lca not None, this node is LCA and return it
    # if LCA found, return it
    if not root:
        return None
    if root == p or root == q:
        return root
    l = self.lowestCommonAncestor(root.left, p, q)
    r = self.lowestCommonAncestor(root.right, p, q)
    if l and r:
        return root  # this node is the LCA
    if l:
        return l
    if r:
        return r
    return None
def lowestCommonAncestorParentPtrs(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    stack = [root]
    parent = {root: None}
    # Iterate until we find both the nodes p and q
    while p not in parent or q not in parent:
        node = stack.pop()
        # While traversing the tree, keep saving the parent pointers.
        if node.left:
            parent[node.left] = node
            stack.append(node.left)
        if node.right:
            parent[node.right] = node
            stack.append(node.right)
    ancestors = set() # all nodes above p in the path to root
    # Process all ancestors for node p using parent pointers.
    while p:
        ancestors.add(p)
        p = parent[p]
    # The first ancestor of q which appears in
    # p's ancestor set() is their lowest common ancestor.
    while q not in ancestors:
        q = parent[q]
    return q

# https://leetcode.com/problems/binary-tree-right-side-view
# level order traversal with O(n) time and space
def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    wl = deque()
    wl.append(root)
    result = []
    while wl:
        levelSize = len(wl)
        for i in range(levelSize):
            node = wl.popleft()
            if i == levelSize - 1:
                result.append(node.val)
            if node.left:
                wl.append(node.left)
            if node.right:
                wl.append(node.right)
    return result
# O(n) time and space
def rightSideViewRecursive(self, root: TreeNode) -> list[int]:
    ans = []
    def dfs(node, level):
        if node:
            if len(ans) < level:
                ans.append(node.val)
            dfs(node.right, level + 1)  # right first
            dfs(node.left, level + 1)  # then left
    dfs(root, 1)
    return ans

# https://leetcode.com/problems/diameter-of-binary-tree/
# O(n) time and space (worst case linked list on call stack)
class SolutionDiameter:
    def __init__(self):
        self.diameter = 0
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        def depth(node):
            left = depth(node.left) if node.left else 0
            right = depth(node.right) if node.right else 0
            self.diameter = max(self.diameter, left + right)
            return 1 + max(left, right)
        depth(root)
        return self.diameter

# https://leetcode.com/problems/sum-root-to-leaf-numbers
class SolutionSumRoot:
    def __init__(self):
        self.total = 0
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def sumNums(node: Optional[TreeNode], num: str):
            if not node.left and not node.right:  # leaf node
                self.total += int(num + str(node.val))
            else:
                if node.left:
                    sumNums(node.left, num + str(node.val))
                if node.right:
                    sumNums(node.right, num + str(node.val))
        sumNums(root, '')
        return self.total
def sumNumbers(self, root: Optional[TreeNode]) -> int:
    def sumNums(node: Optional[TreeNode], num: int):
        if not node:
            return 0
        if not node.left and not node.right:  # leaf node
            return num * 10 + node.val
        else:
            return sumNums(node.left, num * 10 + node.val) + sumNums(node.right, num * 10 + node.val)
    return sumNums(root, 0)

# https://leetcode.com/problems/binary-tree-inorder-traversal
def inorderTraversalIterative(self, root: Optional[TreeNode]) -> List[int]:
    result = []
    if not root:
        return result
    # start as far down left as you can, pop from stack working back upward and move right if possible (and down to left of that new subtree) as you return up
    stack = [root]
    cur = root
    while cur.left:
        stack.append(cur.left)
        cur = cur.left
    while stack:
        cur = stack.pop()
        result.append(cur.val)
        if cur.right:
            stack.append(cur.right)
            temp = cur.right
            while temp.left:
                stack.append(temp.left)
                temp = temp.left
    return result
# https://leetcode.com/problems/binary-search-tree-iterator
# turn the iterative in-order traversal into iterator
class BSTIterator:
    def __init__(self, root: Optional[TreeNode]):
        self.stack = [root]
        cur = root
        while cur.left:
            self.stack.append(cur.left)
            cur = cur.left
    def next(self) -> int:
        if self.stack:
            cur = self.stack.pop()
            result = cur.val
            if cur.right:
                self.stack.append(cur.right)
                temp = cur.right
                while temp.left:
                    self.stack.append(temp.left)
                    temp = temp.left
            return result
        else:
            return -1
    def hasNext(self) -> bool:
        return len(self.stack) > 0
class BSTIterator:
    def __init__(self, root: Optional[TreeNode]):
        self.generator = self._inorder_traversal(root)
        self.next_val = next(self.generator, None)

    def _inorder_traversal(self, node: Optional[TreeNode]):
        if node:
            yield from self._inorder_traversal(node.left)
            yield node.val
            yield from self._inorder_traversal(node.right)

    def next(self) -> int:
        if self.next_val is None:
            return -1
        result = self.next_val
        self.next_val = next(self.generator, None)
        return result

    def hasNext(self) -> bool:
        return self.next_val is not None

# https://leetcode.com/problems/binary-tree-maximum-path-sum
# O(n) time and space (worst case linked list)
def __init__(self):
    self.maxSum = float('-inf')
def maxPathSum(self, root: Optional[TreeNode]) -> int:
    def pathSum(cur):
        if cur:
            left = max(cur.val, cur.val + pathSum(cur.left))
            right = max(cur.val, cur.val + pathSum(cur.right))
            # consider the largest path sum passing through current node (count both children)
            self.maxSum = max(self.maxSum, left - cur.val + right)  # can only count cur.val once
            # return the largest path sum in one of the node's children
            return max(left, right)
        return 0
    pathSum(root)
    return self.maxSum

def maxPathSum2(self, root: Optional[TreeNode]) -> int:
    def pathSum(node):
        if node:
            left = pathSum(node.left)
            right = pathSum(node.right)
            if left < 0: left = 0  # can ignore these leaf paths if they don't help make a larger sum
            if right < 0: right = 0
            self.maxSum = max(self.max, left + node.val + right)  # path sum passing through current node
            return node.val + max(left, right)  # return the path sum picking largest single child path
        return 0
    pathSum(root)
    return self.maxSum

# https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree
def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
    nodes = []  # tuples of (x, y, val)
    def traverse(node, x, y):
        if node:
            nodes.append((x, y, node.val))
            traverse(node.left, x - 1, y + 1)
            traverse(node.right, x + 1, y + 1)

    traverse(root, 0, 0)
    nodes.sort()
    result = []
    curCol = []
    prevX = nodes[0][0]
    for n in nodes:
        x, y, val = n
        if prevX != x:
            result.append(list(curCol))
            curCol.clear()
        curCol.append(val)
        prevX = x
    if curCol:
        result.append(curCol)
    return result
def __init__(self):
    self.minX = float('inf')
    self.maxX = float('-inf')
def verticalTraversalOrig(self, root: Optional[TreeNode]) -> List[List[int]]:
    colToVals = {}  # map y -> [values at this col]

    def traverse(node, x, y):
        if node:
            self.minX = min(self.minX, x)
            self.maxX = max(self.maxX, x)
            if x in colToVals:
                colToVals[x].append((y, node.val))  # need the y values to break tie between numbers at same spot
            else:
                colToVals[x] = [(y, node.val)]
            traverse(node.left, x - 1, y + 1)
            traverse(node.right, x + 1, y + 1)

    traverse(root, 0, 0)
    result = []
    for i in range(self.minX, self.maxX + 1):
        colToVals[i].sort()  # sort by y first and value second to break ties at same location
        result.append([j[1] for j in colToVals[i]])
    return result

# https://leetcode.com/problems/path-sum-ii
def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
    result = []

    def pathSumHelper(cur, curSum, curPath):
        if cur:
            if not cur.left and not cur.right:  # at a leaf
                if curSum + cur.val == targetSum:  # found a desired path
                    curPath.append(cur.val)
                    result.append(list(curPath))
                    curPath.pop()
            else:
                # explore left and right children
                curPath.append(cur.val)
                pathSumHelper(cur.left, curSum + cur.val, curPath)
                pathSumHelper(cur.right, curSum + cur.val, curPath)
                curPath.pop()

    pathSumHelper(root, 0, [])
    return result

# https://leetcode.com/problems/binary-tree-level-order-traversal
def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    result = []
    if not root:
        return result
    wl = deque([root])
    while wl:
        size = len(wl)
        level = []
        for _ in range(size):
            cur = wl.popleft()
            level.append(cur.val)
            if cur.left:
                wl.append(cur.left)
            if cur.right:
                wl.append(cur.right)
        result.append(level)
    return result

# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    while True:
        if root.val > p.val and root.val > q.val:
            root = root.left
        elif root.val < p.val and root.val < q.val:
            root = root.right
        else:
            return root
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if root:
        if root == p:
            return p
        elif root == q:
            return q
        else:
            # use lowestCommonAncestor as a search/find function first, if find p and q then this node is the LCA
            left = self.lowestCommonAncestor(root.left, p, q)
            right = self.lowestCommonAncestor(root.right, p, q)
            if left and right:
                return root
            elif left:  # if result has already been found below, bubble it up
                return left
            elif right:
                return right
    return None

# https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
'''
      1
    2   3
  4      5 
tree as array: [ ,1,2,3,4, , ,5]
indexes:       [0,1,2,3,4,5,6,7]
left child of a node = 2*i
right child of a node = 2*i+1
'''
class Codec:
    # dfs pre-order traversal (originally consider level-order but hard to get a full tree with empty values for missing nodes)
    def serialize(self, root):
        if root:
            return f',{root.val}' + self.serialize(root.left) + ',' + self.serialize(root.right)
        return ''
    # use the index formula to recursively build tree
    def deserialize(self, data):
        parts = data.split(',')
        if len(parts) < 2:
            return None
        def deserial(i):
            if i >= len(parts):
                return None
            return TreeNode(parts[i], deserial(2*i), deserial(2*i+1))
        root = TreeNode(parts[1])
        root.left = deserial(2)
        root.right = deserial(3)
        return root
# c = Codec()
# ser = c.serialize(TreeNode(1,TreeNode(2,TreeNode(4)),TreeNode(3,None,TreeNode(5))))
# print(ser)
# deser = c.deserialize(',1,2,3,4,,,5,')
# print(deser)
# Serialization
class Codec:
    # O(n) time and space
    def deserialize(self, data):
        def rdeserialize(l):
            if l[0] == 'None':
                l.pop(0)
                return None

            root = TreeNode(l[0])
            l.pop(0)
            root.left = rdeserialize(l)
            root.right = rdeserialize(l)
            return root

        data_list = data.split(',')
        root = rdeserialize(data_list)
        return root
    # O(n) time and space
    def serialize(self, root):
        def rserialize(root, string):
            if root is None:
                string += 'None,'
            else:
                string += str(root.val) + ','
                string = rserialize(root.left, string)
                string = rserialize(root.right, string)
            return string
        return rserialize(root, '')

# https://leetcode.com/problems/minimum-distance-between-bst-nodes
def minDiffInBST(self, root: Optional[TreeNode]) -> int:
    # in-order traversal saving the previous and checking the diff
    minDiff = 10 ** 6
    prev = -10 ** 6

    def minDiffBST(node):
        nonlocal minDiff, prev
        if node:
            minDiffBST(node.left)
            minDiff = min(minDiff, node.val - prev)
            prev = node.val
            # print('checking minDiff', minDiff, 'setting prev to', prev)
            minDiffBST(node.right)

    minDiffBST(root)
    return minDiff

# https://leetcode.com/problems/balance-a-binary-search-tree
# O(n) time and space
def balanceBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    nodes = [] # put all nodes in sorted order here
    def inorder(n):
        if n:
            inorder(n.left)
            nodes.append(n)
            inorder(n.right)
            n.left = None
            n.right = None

    def build(lo, hi):
        if lo <= hi:
            i = (lo + hi) // 2
            # print('adding i', i, 'to tree between lo', lo, 'and high', hi)
            cur = nodes[i]
            cur.left = build(lo, i - 1)
            cur.right = build(i + 1, hi)
            return cur
        return None

    inorder(root)
    return build(0, len(nodes) - 1)

# https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves
# same as https://leetcode.com/problems/smallest-subtree-with-all-the-deepest-nodes
def subtreeWithAllDeepest(self, root):
    # return node, depth pairs
    def dfs(node):
        if not node: return None, 0
        # process dfs in post-order
        L, R = dfs(node.left), dfs(node.right)
        # if cur node not a leaf or balanced node, return the previously found node with the larger depth
        if L[1] > R[1]: return L[0], L[1] + 1
        if L[1] < R[1]: return R[0], R[1] + 1
        # it's a leaf node or a balanced depth node, it could be the answer, return itself and its subtree depth
        return node, L[1] + 1
    return dfs(root)[0]
def subtreeWithAllDeepest(self, root):
    # Tag each node with it's depth.
    depth = {}
    def dfs(node, d):
        if node:
            depth[node] = d
            dfs(node.left, d + 1)
            dfs(node.right, d + 1)
    dfs(root, 0)
    max_depth = max(depth.values())
    def answer(node):
        # Return the answer for the subtree at node.
        if not node or depth.get(node, -1) == max_depth:
            return node
        L, R = answer(node.left), answer(node.right)
        return node if L and R else L or R
    return answer(root)
def lcaDeepestLeavesUgly(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    def lcaMaxLeaves(node):
        if not node:
            return 0
        if node in self.deepest:
            if 1 == len(self.deepest):
                self.result = node
            return 1
        numMaxLeaves = lcaMaxLeaves(node.left) + lcaMaxLeaves(node.right)
        # print('at node', node.val, 'nummaxleaves', numMaxLeaves)
        if numMaxLeaves == len(self.deepest) and not self.result:
            # print('overwriting self.result')
            self.result = node
            # return float('-inf')
        return numMaxLeaves

    def maxDepth(node, d):
        if node:
            self.maxDepth = max(self.maxDepth, d)
            maxDepth(node.left, d + 1)
            maxDepth(node.right, d + 1)

    def saveMaxLeaves(node, d):
        if node:
            if d == self.maxDepth:
                self.deepest.add(node)
            saveMaxLeaves(node.left, d + 1)
            saveMaxLeaves(node.right, d + 1)

    maxDepth(root, 0)
    # print('max depth', self.maxDepth)
    saveMaxLeaves(root, 0)
    # print('deepest', self.deepest)
    lcaMaxLeaves(root)
    return self.result

# https://leetcode.com/problems/count-good-nodes-in-binary-tree
# iterative BFS: O(n) time and space
def goodNodes(self, root: TreeNode) -> int:
    result = 0
    q = deque([(root, root.val)])
    while q:
        cur, pathMax = q.popleft()
        if pathMax <= cur.val:
            result += 1
        if cur.left:
            q.append((cur.left, max(cur.val, pathMax)))
        if cur.right:
            q.append((cur.right, max(cur.val, pathMax)))
    return result
# recursive dfs: O(n) time and space
def goodNodes(self, root: TreeNode) -> int:
    result = 0
    def dfs(node, pathMax):
        nonlocal result
        if node:
            if pathMax <= node.val:
                result += 1
            dfs(node.left, max(node.val, pathMax))
            dfs(node.right, max(node.val, pathMax))
    dfs(root, root.val)
    return result

# https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree
# O(n) time and space
def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
    parents = {}
    result = []
    visited = set()

    def saveParents(c, p):
        if c:
            parents[c] = p
            saveParents(c.left, c)
            saveParents(c.right, c)

    def explore(c, d):
        if c and c not in visited:
            visited.add(c)
            if d == k:
                result.append(c.val)
            else:
                explore(c.left, d + 1)
                explore(c.right, d + 1)
                explore(parents[c], d + 1)

    saveParents(root, None)
    explore(target, 0)
    return result

# https://leetcode.com/problems/check-completeness-of-a-binary-tree
# do a BFS: tree is complete if there is no node to the right of the first null node and no node at a greater level than the first null node
def isCompleteTreeBest(self, root: Optional[TreeNode]) -> bool:
    seenNull = False
    q = deque([root])
    while q:
        cur = q.popleft()
        if not cur:
            seenNull = True
        else:
            if seenNull:
                return False
            q.append(cur.left)  # append None values in next row
            q.append(cur.right)
    return True
def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
    prev = None
    final = False
    expectedSize = 1
    q = deque([root])
    while q:
        levelSize = len(q)
        if levelSize != expectedSize:
            final = True
        expectedSize *= 2
        for _ in range(levelSize):
            cur = q.popleft()
            if cur:
                if cur != root and not prev: return False  # level has a gap in it
                q.append(cur.left)  # append None values in next row
            if cur:
                if cur != root and not prev: return False
                q.append(cur.right)
            prev = cur
        if final and q:  # wasn't a full level so should have been last and wasn't
            return False
    return True

