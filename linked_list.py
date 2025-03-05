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

# https://leetcode.com/problems/intersection-of-two-linked-lists
#   same as https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii
def getIntersectionNodeUgly(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
    # go until one is past root, then reassign it to the other starting point and continue until 2nd one
    # is past root, set it to the opposite starting location and proceed until they meet
    n1 = headA
    n2 = headB
    if n1 == n2:
        return n1
    while n1 and n2:
        n1 = n1.next
        n2 = n2.next
    if n1:
        n2 = headA
    else:
        n1 = headB

    while n1 and n2:
        n1 = n1.next
        n2 = n2.next
    if n1:
        n2 = headA
    else:
        n1 = headB

    while n1 and n2:
        if n1 == n2:
            return n1
        n1 = n1.next
        n2 = n2.next
    return None

# https://leetcode.com/problems/remove-nth-node-from-end-of-list
def removeNthFromEnd(self, head, n):
    dummy = ListNode(0)
    dummy.next = head
    first = dummy
    second = dummy
    # Advances first pointer so that the gap between first and second is n nodes apart
    for i in range(n + 1):
        first = first.next
    # Move first to the end, maintaining the gap
    while first is not None:
        first = first.next
        second = second.next
    second.next = second.next.next
    return dummy.next

# https://leetcode.com/problems/sum-of-two-integers
def getSum(self, a: int, b: int) -> int:
    x, y = abs(a), abs(b)
    # ensure that abs(a) >= abs(b)
    if x < y:
        return self.getSum(b, a)

    # abs(a) >= abs(b) -->
    # a determines the sign
    sign = 1 if a > 0 else -1

    if a * b >= 0:
        # sum of two positive integers x + y
        # where x > y
        while y:
            answer = x ^ y
            carry = (x & y) << 1
            x, y = answer, carry
    else:
        # difference of two integers x - y
        # where x > y
        while y:
            answer = x ^ y
            borrow = ((~x) & y) << 1
            x, y = answer, borrow

    return x * sign
def getSum(self, a: int, b: int) -> int:
    mask = 0xFFFFFFFF

    while b != 0:
        a, b = (a ^ b) & mask, ((a & b) << 1) & mask

    max_int = 0x7FFFFFFF
    return a if a < max_int else ~(a ^ mask)
