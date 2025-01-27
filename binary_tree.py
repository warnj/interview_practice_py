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