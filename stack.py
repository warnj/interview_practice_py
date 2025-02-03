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
