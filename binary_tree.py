from collections import deque
from typing import Optional, List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

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
