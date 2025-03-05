's'.isalpha()  # alphabetic, isalnum() also includes numbers
'1'.isnumeric() # isdigit() also works
ord('a')  # unicode code ordinal = 97
chr(97)  # character from the ascii/unicode = 'a'

# String (immutable)
'ABC'.lower()  # lowercase
"hi there".split()  # split on whitespace or optional delimiter param
" whitespace\t".strip()  # remove leading and trailing whitespace
"hi there".count("h")  # count substring occurences
"hi there".startswith("hi")
sorted("to sort")  # result=[' ', 'o', 'o', 'r', 's', 't', 't']   sort chars in string, return list
''.join(['a','b','c'])  # result='abc'  turn a list into string (use list as a StringBuilder)
print('he' in 'hero', 'x' not in 'hero')
'hi there'.find('t', 0)  # return -1 if not found, .index() will do same thing but raise ValueError if not found
'hi there'.rfind('e')  # return highest index given substring is found
list("abcd")  # string to list (can also convert to tuple)
"reverse me"[::-1]

# Search
import bisect
bisect.bisect_left([1,4,7,9], 4) # binary search - returns first index where x would be inserted to keep sorted order
# bisect_right - returns the last index where x would be inserted to keep sorted order
[1, 2, 3, 4].index(3)  # returns the first index of 3 (2 in this case) or ValueError if not found
found = 3 in [1, 2, 3, 4]

# Numbers
tiny = float('-inf')
huge = float('inf')
import math
ceil = math.ceil(2.5)  # smallest integer >= x
four = math.sqrt(16)
intQuotient = 5 // 2
tenPow9 = 10 ** 9  # 10^9
five = abs(-5)  # also min(), max(), sum(), pow(), etc
round(3.14159, 2)  # Output: 3.14

# Boolean
all(['True', 'True'])
any(['True', 'False'])

# Loop
for i in range(start=0, stop=2, step=1):  # i=start; i<stop; i+=step
    break  # exits the innermost loop only
    continue  # continues the innermost loop
else:
    pass # pass does nothing, else executes if for loop completes without hitting a break statement

# Tuple (immutable)
sort([(1,3),(1,2)])  # [(1,2),(1,3)] naturally ordered by breaking ties with next item
tuple([1,2,3])  # list to tuple (also works with set)
list((1,2,3)) # tuple to list
frozenset([2,3]) # immutable set (good for hashing like a tuple)
for name, age in zip(['ava','bob'], [22, 31]):  # combine iterables into tuple pairs stopping at end of shortest one
    print(f'name {name}, age {age}')

# List (mutable, closest thing to array)
array = [0] * 10  # 10 zeros
arrayWithoutLast4 = array[:-4]
shallowCopy = list(array)
newList = [1] + [2,3,4]
del newList[2] # remove an element
import copy
deepCopy = copy.deepcopy(array)
for i, n in enumerate([9,8,7,6]):
    print(f"Index: {i}, Value: {n}")
[2, 3].insert(0, 1)  # put 1 at index 0
array.extend([5,6,7])  # add all elements to array
newRev = reversed([1,2,3])  # [1,2].reverse() also works and modifies list without returning
t = tuple([1,2,3])  # list to tuple (also works with set)
l = list((1,2,3))  # tuple to list
[1,2,3].clear()  # set to empty list
from sortedcontainers import SortedList
sList = SortedList([5,1])
bisect.bisect_left(sList, 5) # O(logn) find
sList.add(4) # O(logn) add
sList.remove(5) # O(logn) remove

# Sort
newList = sorted([[2,3],[9,8],[4,5]], key=lambda x: (x[0], x[1]))  # sort by 1st element, then 2nd
[[2],[9,8],[4,5]].sort(key=lambda x: x[0], reverse=True)  # sort by the first element, order in reverse
newList = sorted(data, key=lambda x: (x[0], x[1] if len(x) > 1 else float('inf')))  # sort by 2nd element if it exists
def char_pair_diff(word):
    return sum(abs(ord(word[i]) - ord(word[i + 1])) for i in range(len(word) - 1))
['ab','xz'].sort(key=char_pair_diff)

# 2-D Arrays
x, y = 4, 5
array2dBools = [[False] * x for _ in range(y)]

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
heapq.heapify(minHeap)  # O(n) runtime!  note: sorts tuples by first element and then by second
top = minHeap[0]
topRemoved = heapq.heappop(minHeap)
heapq.heappush(minHeap, 6)
heapq.heappushpop(minHeap, 6) # add given element, then remove top of heap

# Set:
s = {1,2,3}
s2 = set([1,2,3])  # list to set (also works with tuple)
s3 = set()
s.add(4)  # adding the same thing would be a noop with no error
s.remove(4)  # KeyError if not found
s.discard(10)  # No KeyError if not found
s.clear()
s.update(s2)  # Adds all elements of s2 to s
b = 3 in s
# other options like Union |, Intersection &, Difference -, Symmetric Difference ^

# Dictionary:
person = {
    "name": "John",
    "age": 30, # can mix value types
}
n1 = person["name"]
n2 = person.get("name", 'default value')  # avoids KeyError if key doesn't exist
person["job"] = "Engineer"
person["age"] = 31
del person["name"]
job = person.pop("job")  # returns the value of the removed key
val = person.setdefault('name', 'default')  # sets person['name'] = 'default' if name doesn't exist
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

# Special Dictionaries:
from collections import defaultdict
counts = defaultdict(int)  # interact with missing elements as though they are present with value of 0
for elem in ['a', 'b', 'a']: counts[elem] += 1
from sortedcontainers import SortedDict
sMap = SortedDict({2:'b', 1:'a', 3:'c'})
idx = sMap.bisect_left(2)  # Output: 1: index of the smallest key >= 2
k, v = sMap.peekitem(idx)  # Output: 2,'b'
idx = sMap.bisect_right(2)  # Output: 2: index of the smallest key > 2
if idx < len(sMap):
    k, v = sMap.peekitem(idx)  # Output: 3,'c'
from collections import OrderedDict
od = OrderedDict()
od.popitem(last=False) # removes first item
od.move_to_end(key, last=False)  # move to Front
from collections import Counter  # any hashable item works in Counter
counter = Counter(['a', 'b', 'a'])  # Counter({'a': 2, 'b': 1})
counter.values()  # [2, 1]

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
random.shuffle([1, 2, 3])

# Object Oriented
hashCode = hash('abc')
class MyClass:
    staticVar = 100
    def __init__(self):
        self.instanceVar = 1
        print(MyClass.staticVar)
    @staticmethod
    def static_func():
        print('static function')
    def _private_function(self):
        print("This is private function naming convention")
class Comparable:
    def __init__(self, value):
        self.value = value
    def __eq__(self, other): # compare objects on content not memory address
        if isinstance(other, Comparable):
            return self.value == other.value
        return NotImplemented
    def __lt__(self, other): # make comparable for sorting
        if isinstance(other, Comparable):
            return self.value < other.value
        return NotImplemented
    def __repr__(self):
        return f"Comparable(value={self.value})"
    def __len__(self):
        return len(self.value)
    def __hash__(self):
        return hash(self.value)  # self.value must be immutable

# Typing
from typing import Optional
# def maybe_return_value(condition: bool) -> Optional[int]:

# Keywords
nonlocal x  # allow reassigning (not needed for mutation) to a variable from outer scope

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
float('2.1')

num = 5  # 0b101
bit_to_check = 1  # check 2nd bit from the right is 1
is_set = (num & (1 << bit_to_check)) != 0  # True if set

bit_to_set = 1  # set 2nd bit from the right to 1
num |= (1 << bit_to_set)  # Result: 7 (0b111)

bit_to_clear = 0  # 1st bit from the right
num &= ~(1 << bit_to_clear)  # Result: 4 (0b100)

bit_to_toggle = 1  # 2nd bit from the right
num ^= (1 << bit_to_toggle)  # Result: 7 (0b111)

# One-liners
[p for p in person.values() if len(p) > 1]
