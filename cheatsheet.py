# Chars
's'.isalpha()  # alphabetic
'1'.isnumeric()
ord('s')  # unicode code ordinal

# Numbers
tiny = float('-inf')
huge = float('inf')
intQuotient = 5 // 2

# Array (list
array = [0] * 10  # 10 zeros
shallowCopy = list(array)
import copy
deepCopy = copy.deepcopy(array)
for i, n in enumerate([9,8,7,6]):
    print(f"Index: {i}, Value: {n}")
# for i in range(start, stop, step):
sorted_lists = sorted([[2,3],[9,8,7],[4,5,6]], key=lambda x: x[0])  # sort by the first element
[[2,3],[9,8,7],[4,5,6]].sort(key=lambda x: x[0], reverse=True)  # sort by the first element, order in reverse

# 2-D Arrays
x, y = 4, 5
array2dBools = [[False for _ in range(x)] for _ in range(y)]

# Set:
s = {1,2,3}
s2 = set([1,2,3])
s3 = set()
s.add(4)
s.remove(4)  # KeyError if not found
s.discard(10)  # No KeyError if not found
s.clear()
b = 3 in s
# other options like Union |, Intersection &, Difference -, Symmetric Difference ^

# Dictionary:
person = {
    "name": "John",
    "age": 30,
}
n1 = person["name"]
n2 = person.get("name")  # avoids KeyError if key doesn't exist
person["job"] = "Engineer"
person["age"] = 31
del person["name"]
job = person.pop("job")  # returns the value of the removed key
person.clear()  # remove all items
print("name" in person)
print("job" not in person)
for key in person:  # Loop through keys
    print(key)
for value in person.values():  # Loop through values
    print(value)
for key, value in person.items():  # Loop through key-value pairs
    print(f"{key}: {value}")

# Stack:
stack = []
stack.append(5)
top = stack[-1]
topRemoved = stack.pop()
isEmpty = not stack
isEmpty2 = len(stack) == 0

# Queue:
# double ended queues have O(1) speed for appendleft() and popleft() while lists have O(n) performance for insert(0, value) and pop(0)
from collections import deque

q = deque()
q.append(4)
front = q.popleft()

# Heap:
import heapq

minHeap = [3,2,1,4,5]
heapq.heapify(minHeap)  # note: sorts tuples by first element and then by second
top = minHeap[0]
topRemoved = heapq.heappop(minHeap)
heapq.heappush(minHeap, 6)


# Binary Tree

# Linked List

# DFS

# BFS

# Random
import random

random.randint(5, 10)  # inclusive start and end
randomFloat = random.random()
randomFloatRange = random.uniform(5.5, 9.5)
randomElem = random.choice([10, 20, 30, 40, 50])

# Object Oriented
class MyClass:
    def _private_function(self):
        print("This is private function naming convention")

# Typing
from typing import Optional
# def maybe_return_value(condition: bool) -> Optional[int]:

# Exceptions

# Regex
