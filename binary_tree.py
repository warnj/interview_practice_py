from collections import deque
from typing import Optional

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
