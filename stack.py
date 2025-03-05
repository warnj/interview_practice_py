from collections import deque
from typing import List

# https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/
# O(n) time and space
def minRemoveToMakeValid(self, s: str) -> str:
    s = list(s)
    stack = []
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
            else:
                s[i] = ''
    while stack:
        s[stack.pop()] = ''
    return ''.join(s)

# https://leetcode.com/problems/daily-temperatures
# O(n) time and space
def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
    result = [0] * len(temperatures)
    stack = []
    for i, t in enumerate(temperatures):
        while stack and t > temperatures[stack[-1]]:
            # current index is the next warmer day for the day on the stack (since we move left to right)
            prevIndex = stack.pop()
            result[prevIndex] = i - prevIndex
        stack.append(i)
    return result

# https://leetcode.com/problems/exclusive-time-of-functions
def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
    result = [0] * n
    starts = []  # [stack of start times old to new]
    for log in logs:
        proc, action, time = log.split(':')
        if action == 'start':
            if starts:  # previously running program will pause for now
                curProc, startTime = starts[-1]
                result[curProc] += int(time) - startTime
            starts.append((int(proc), int(time)))
        else:  # end
            curProc, startTime = starts.pop()
            result[curProc] += int(time) - startTime + 1
            if starts:
                starts[-1] = (starts[-1][0], int(time) + 1)  # previous program in stack resumes, save new start
    return result

# https://leetcode.com/problems/implement-queue-using-stacks
class MyQueue:
    def __init__(self):
        self.s1 = [] # front at the top
        self.s2 = [] # end at the top
    def push(self, x: int) -> None: # O(1)
        self.s2.append(x)
    def pop(self) -> int:  # O(1) amortized O(n) worst
        if not self.s1:
            while self.s2:
                self.s1.append(self.s2.pop())
        return self.s1.pop()
    def peek(self) -> int:  # O(1) amortized O(n) worst
        if not self.s1:
            while self.s2:
                self.s1.append(self.s2.pop())
        return self.s1[-1]
    def empty(self) -> bool:
        return len(self.s1) == 0 and len(self.s2) == 0
class MyQueueEasy:
    def __init__(self):
        self.s = []
    def push(self, x: int) -> None: # O(n)
        # add to front
        temp = []
        while self.s:
            temp.append(self.s.pop())
        self.s.append(x)
        while temp:
            self.s.append(temp.pop())
    def pop(self) -> int: # O(1)
        return self.s.pop()
    def peek(self) -> int: # O(1)
        return self.s[-1]
    def empty(self) -> bool:
        return len(self.s) == 0

# https://leetcode.com/problems/implement-stack-using-queues/
# variation of this exists with single queue where you do len(q)-1 rotations (pop and push) after a new push so back of queue always has oldest element
class MyStackEasy:
    def __init__(self):
        # remove from top, add to top
        self.q1 = deque() # store in order

    def push(self, x: int) -> None:
        self.q1.append(x)

    def pop(self) -> int:
        q2 = deque()
        last = 0
        while self.q1:
            last = self.q1.popleft()
            if self.q1:
                q2.append(last)
        self.q1 = q2
        return last

    def top(self) -> int:
        return self.q1[-1]

    def empty(self) -> bool:
        return len(self.q1) == 0
