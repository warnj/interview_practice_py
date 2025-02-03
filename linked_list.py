import heapq
from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# https://leetcode.com/problems/add-two-numbers
def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    resultHead = None
    resultLast = None
    carry = 0
    cur1 = l1
    cur2 = l2

    while cur1 or cur2 or carry:
        n1 = cur1.val if cur1 else 0
        n2 = cur2.val if cur2 else 0
        total = n1 + n2 + carry
        if total > 9:
            carry = 1
        else:
            carry = 0

        new = ListNode(total % 10)
        if resultLast:
            resultLast.next = new
            resultLast = resultLast.next
        else:
            resultHead = new
            resultLast = new

        if cur1:
            cur1 = cur1.next
        if cur2:
            cur2 = cur2.next
    return resultHead

# https://leetcode.com/problems/merge-k-sorted-lists
def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    class HeapVal:
        def __init__(self, v, n=None):
            self.val = v
            self.node = n
        def __eq__(self, other):
            if isinstance(other, HeapVal):
                return self.val == other.val
            return NotImplemented
        def __lt__(self, other):
            if isinstance(other, HeapVal):
                return self.val < other.val
            return NotImplemented

    heap = []
    for l in lists:
        if l:
            heap.append(HeapVal(l.val, l))
    heapq.heapify(heap)

    result = ListNode(-1)
    resultEnd = result

    while heap:
        v = heapq.heappop(heap)
        resultEnd.next = ListNode(v.val)
        resultEnd = resultEnd.next
        if v.node.next:
            heapq.heappush(heap, HeapVal(v.node.next.val, v.node.next))

    return result.next