# Chars
's'.isalpha()  # alphabetic
'1'.isnumeric()
ord('a')  # unicode code ordinal = 97
chr(97)  # character from the ascii/unicode = 'a'

# String (immutable)
'ABC'.lower()  # lowercase
"hi there".split()  # split on whitespace or optional delimiter param
"hi there".count("h")  # count substring occurences
"hi there".startswith("hi")
''.join(sorted("chars to sort"))  # sort chars in string
print('hero' in 'superhero')
'hi there'.find('t')  # return -1 if not found, .index() will do same thing but raise ValueError if not found
'hi there'.rfind('e')
list("abcd")  # string to list (can also convert to tuple)

# Numbers
tiny = float('-inf')
huge = float('inf')
intQuotient = 5 // 2
tenPow9 = 10 ** 9
five = abs(-5)  # also min(), max(), sum(), pow(), etc
round(3.14159, 2)  # Output: 3.14

# Boolean
all(['True', 'True'])
any(['True', 'False'])

# Loop
for i in range(start=0, stop=2, step=1):  # i=start; i<stop; i+=step
    break  # exits the innermost loop only
    continue  # continues the innermost loop

# Tuple (immutable)
sort([(1,3),(1,2)])  # [(1,2),(1,3)] naturally ordered by breaking ties with next item
tuple([1,2,3])  # list to tuple (also works with set)
list((1,2,3)) # tuple to list

# List (mutable, closest thing to array)
array = [0] * 10  # 10 zeros
arrayWithoutLast4 = array[:-4]
shallowCopy = list(array)
newList = [[1] + [2,3,4]]
import copy
deepCopy = copy.deepcopy(array)
for i, n in enumerate([9,8,7,6]):
    print(f"Index: {i}, Value: {n}")
array.extend([5,6,7])  # add all elements to array
sorted_lists = sorted([[2,3],[9,8,7],[4,5,6]], key=lambda x: x[0])  # sort by the first element
[[2,3],[9,8,7],[4,5,6]].sort(key=lambda x: x[0], reverse=True)  # sort by the first element, order in reverse
reversed([1,2,3])

# 2-D Arrays
x, y = 4, 5
array2dBools = [[False for _ in range(x)] for _ in range(y)]

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
q.appendleft(5)
front = q.popleft()

# Heap:
import heapq
minHeap = [3,2,1,4,5]
heapq.heapify(minHeap)  # note: sorts tuples by first element and then by second
top = minHeap[0]
topRemoved = heapq.heappop(minHeap)
heapq.heappush(minHeap, 6)

# Set:
s = {1,2,3}
s2 = set([1,2,3])  # list to set (also works with tuple)
s3 = set()
s.add(4)  # adding the same thing would be a noop with no error
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
n2 = person.get("name", 'default value')  # avoids KeyError if key doesn't exist
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
# Unhashable types (mutable): dict, list, set – use tuple or frozenset

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
hashCode = hash('abc')
class MyClass:
    def _private_function(self):
        print("This is private function naming convention")
class Comparable:
    def __init__(self, value):
        self.value = value
    def __eq__(self, other):
        if isinstance(other, Comparable):
            return self.value == other.value
        return NotImplemented
    def __lt__(self, other):
        if isinstance(other, Comparable):
            return self.value < other.value
        return NotImplemented
    def __repr__(self):
        return f"Comparable(value={self.value})"

# Typing
from typing import Optional
# def maybe_return_value(condition: bool) -> Optional[int]:

# Exceptions

# Regex
import re
re.split(r'/+', "/usr//mnt/e")  # ['','usr','mnt','e'] raw string regex to match single & multiple slashes
re.findall(r"\d+", "123abc456")  # ["123","456"] match digits

# Print Debugging
print(f"x: {x}")

# Binary / Bit Operations
bin(5)  # '0b101'
int('101', 2)  # 5

num = 5  # 0b101
bit_to_check = 1  # check 2nd bit from the right is 1
is_set = (num & (1 << bit_to_check)) != 0  # True if set

bit_to_set = 1  # set 2nd bit from the right to 1
num |= (1 << bit_to_set)  # Result: 7 (0b111)

bit_to_clear = 0  # 1st bit from the right
num &= ~(1 << bit_to_clear)  # Result: 4 (0b100)

bit_to_toggle = 1  # 2nd bit from the right
num ^= (1 << bit_to_toggle)  # Result: 7 (0b111)
