# https://leetcode.com/problems/longest-substring-without-repeating-characters/
# two pointer: O(n) time and O(1) space (max set size limited by # of different chars)
def lengthOfLongestSubstring(self, s: str) -> int:
    lo = 0
    hi = 0
    chars = set()
    result = 0
    while hi < len(s):
        if s[hi] in chars:
            # move lo to right
            while s[hi] in chars:
                chars.remove(s[lo])
                lo += 1
        else:
            # move hi to right
            chars.add(s[hi])
            result = max(result, hi - lo + 1)
            hi += 1
    return result
