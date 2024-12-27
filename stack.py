
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
