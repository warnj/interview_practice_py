# https://leetcode.com/problems/longest-substring-without-repeating-characters/
# two pointer: O(n) time and O(1) space (max set size limited by # of different chars)
import re
from collections import deque
from typing import List


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

# https://leetcode.com/problems/group-anagrams
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    # represent words as sorted strings or as int[26] of char counts
    def sortedChars(s):
        return ''.join(sorted(s))
    buckets = {}  # sorted char str -> [strings with that sorted char]
    for s in strs:
        chars = sortedChars(s)
        if chars in buckets:
            buckets[chars].append(s)
        else:
            buckets[chars] = [s]
    result = []
    for l in buckets.values():
        result.append(l)
    return result
def groupAnagrams2(self, strs: List[str]) -> List[List[str]]:
    def countChars(s):
        counts = [0] * 26
        for c in s:
            counts[ord(c) - ord('a')] += 1
        return tuple(counts)
    buckets = {}  # tuple of char counts -> [strings with the counts]
    for s in strs:
        counts = countChars(s)
        if counts in buckets:
            buckets[counts].append(s)
        else:
            buckets[counts] = [s]
    result = []
    for l in buckets.values():
        result.append(l)
    return result

# https://leetcode.com/problems/count-ways-to-build-good-strings
def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
    # Use dp[i] to record to number of good strings of length i.
    dp = [1] + [0] * (high)
    mod = 10 ** 9 + 7
    # Iterate over each length `end`.
    for end in range(1, high + 1):
        # check if the current string can be made by append zero `0`s or one `1`s.
        if end >= zero:
            dp[end] += dp[end - zero]
        if end >= one:
            dp[end] += dp[end - one]
        dp[end] %= mod
    # Add up the number of strings with each valid length [low ~ high].
    return sum(dp[low: high + 1]) % mod

# https://leetcode.com/problems/maximum-score-after-splitting-a-string
# O(n) time and O(1) space
def maxScore(self, s: str) -> int:
    z = 1 if s[0] == '0' else 0  # number of 0s in first part of string
    o = 0  # number of 1s in last part of string
    for i in range(1, len(s)):
        if s[i] == '1':
            o += 1
    divider = 1  # first index of the last part of string
    maxScore = z + o
    while divider < len(s) - 1:
        if s[divider] == '0':
            z += 1
        else:
            o -= 1
        divider += 1
        maxScore = max(maxScore, z + o)
    return maxScore

# https://leetcode.com/problems/minimum-number-of-operations-to-move-all-balls-to-each-box
# O(n) time and O(1) space
def minOperations(self, boxes: str) -> List[int]:
    ans = [0] * len(boxes)
    boxesToLeft = 0
    movesToLeft = 0
    for i in range(len(boxes)):
        ans[i] += movesToLeft
        if boxes[i] == '1':
            boxesToLeft += 1
        movesToLeft += boxesToLeft
    boxesToRight = 0
    movesToRight = 0
    for i in range(len(boxes) - 1, -1, -1):
        ans[i] += movesToRight
        if boxes[i] == '1':
            boxesToRight += 1
        movesToRight += boxesToRight
    return ans

# https://leetcode.com/problems/shifting-letters-ii
# O(n+m) time and O(n) space
def shiftingLetters(self, s: str, shifts: list[list[int]]) -> str:
    n = len(s)
    diff_array = [0] * n # diff[i] is the difference in the number of shifts for i than i-1
    for shift in shifts:
        if shift[2] == 1:  # If direction is forward (1)
            diff_array[shift[0]] += 1  # Increment at the start index
            if shift[1] + 1 < n:
                diff_array[shift[1] + 1] -= 1  # Decrement at the end+1 index
        else:  # If direction is backward (0)
            diff_array[shift[0]] -= 1  # Decrement at the start index
            if shift[1] + 1 < n:
                diff_array[shift[1] + 1] += 1  # Increment at the end+1 index
    result = list(s)
    number_of_shifts = 0
    for i in range(n):
        number_of_shifts = (number_of_shifts + diff_array[i]) % 26  # Update cumulative shifts, keeping within the alphabet range
        if number_of_shifts < 0:
            number_of_shifts += 26  # Ensure non-negative shifts
        # Calculate the new character by shifting `s[i]`
        shifted_char = chr((ord(s[i]) - ord("a") + number_of_shifts) % 26 + ord("a"))
        result[i] = shifted_char
    return "".join(result)

# https://leetcode.com/problems/string-matching-in-an-array
# O(n^2 * m^2) time
# alternative advanced solutions: suffix trie
def stringMatching(self, words: List[str]) -> List[str]:
    r = []
    for i, w in enumerate(words):
        for j in range(len(words)):
            if i != j and w in words[j]:
                r.append(w)
                break
    return r

# https://leetcode.com/problems/longest-palindromic-substring
def longestPalindrome(self, s: str) -> str:
    def palindromeLenOdd(s, i):
        r = deque()
        r.append(s[i])
        lo = i - 1
        hi = i + 1
        while lo >= 0 and hi < len(s) and s[lo] == s[hi]:
            r.append(s[hi])
            r.appendleft(s[lo])
            lo -= 1
            hi += 1
        return ''.join(r)

    def palindromeLenEven(s, lo, hi):
        r = deque()
        while lo >= 0 and hi < len(s) and s[lo] == s[hi]:
            r.append(s[hi])
            r.appendleft(s[lo])
            lo -= 1
            hi += 1
        return ''.join(r)

    result = s[0]
    for i in range(1, len(s)):
        o = palindromeLenOdd(s, i)
        if len(o) > len(result):
            result = o
        e = palindromeLenEven(s, i - 1, i)
        if len(e) > len(result):
            result = e
    return result

# https://leetcode.com/problems/unique-length-3-palindromic-subsequences
# O(26*n) time and O(26) space
def countPalindromicSubsequence(self, s: str) -> int:
    r = 0
    for i in range(26):
        c = chr(ord('a') + i)
        # find first and last of each possible letter, everything in between is a palindrome of len 3
        lo = s.find(c)
        if lo < 0:
            continue
        hi = s.rfind(c)
        if hi > lo + 1:
            uniqueLettersBetween = set()
            for mid in range(lo + 1, hi):
                uniqueLettersBetween.add(s[mid])
            r += len(uniqueLettersBetween)
    return r
# O(n^3) time and O(26*26) space
def countPalindromicSubsequenceBrute(self, s: str) -> int:
    strs = set()
    for i in range(len(s) - 2):
        for j in range(i + 1, len(s) - 1):
            for k in range(j + 1, len(s)):
                if s[i] == s[k]:
                    strs.add(s[i] + s[j] + s[k])
    return len(strs)

# https://leetcode.com/problems/valid-palindrome
def isPalindrome(self, s: str) -> bool:
    lo = 0
    hi = len(s) - 1
    s = s.lower()
    while lo < hi:
        while lo < hi and not s[lo].isalpha() and not s[lo].isnumeric():
            lo += 1
        while lo < hi and not s[hi].isalpha() and not s[hi].isnumeric():
            hi -= 1
        if s[lo] != s[hi]:
            return False
        lo += 1
        hi -= 1
    return True

# https://leetcode.com/problems/simplify-path
def simplifyPath(self, path: str) -> str:
    parts = re.split(r'/+', path)
    result = []
    for part in parts:
        if part == '..':
            if result:
                result.pop()
        elif part != '.':
            result.append(part)
    result = '/'.join(result)
    if not result.startswith('/'):
        result = '/' + result
    if len(result) > 1 and result.endswith('/'):
        result = result[:-1]
    return result

# https://leetcode.com/problems/valid-palindrome-ii
def validPalindrome(self, s: str) -> bool:
    def isPalindrome(lo, hi, hasSkipped):
        while lo < hi:
            if s[lo] != s[hi]:
                if hasSkipped:
                    return False  # Cannot skip two characters
                # Try skipping one character from either end
                return isPalindrome(lo + 1, hi, True) or isPalindrome(lo, hi - 1, True)
            lo += 1
            hi -= 1
        return True
    return isPalindrome(0, len(s) - 1, False)

# https://leetcode.com/problems/construct-k-palindrome-strings
# O(n) time and O(26) space
def canConstruct(self, s: str, k: int) -> bool:
    # chars that occur odd numbers of times must occupy the middle of a palindrome
    if k > len(s):
        return False
    counts = [0] * 26
    for char in s:
        counts[ord(char) - ord('a')] += 1
    for c in counts:
        if c % 2 == 1:
            k -= 1
            if k < 0:  # cant make enough palindromes to use up the odd numbered chars
                return False
    return True

# https://leetcode.com/problems/minimum-length-of-string-after-operations
# O(n) time and O(26) space
def minimumLength(self, s: str) -> int:
    # counts of 3, 5, 7 -> 1
    # counts of 1 -> 1
    # counts of 2 -> 2
    # counts of 4, 6, 8 -> 2
    counts = [0] * 26
    for char in s:
        counts[ord(char) - ord('a')] += 1
    total = 0
    for count in counts:
        if count == 1 or count == 2:
            total += count
        elif count % 2 == 1:
            total += 1
        elif count > 0:
            total += 2
    return total

# https://leetcode.com/problems/word-break
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    word_set = set(wordDict)
    memo = {}
    def explore(s):
        if s in memo:
            return memo[s]
        if s in word_set:
            return True
        for i in range(1, len(s)):
            if s[:i] in word_set and explore(s[i:]):
                return True
        memo[s] = False
        return False
    return explore(s)
def wordBreakBetter(self, s: str, wordDict: List[str]) -> bool:
    dp = [False] * len(s)
    for i in range(len(s)):
        for word in wordDict:
            # Handle out of bounds case
            if i < len(word) - 1:
                continue
            if i == len(word) - 1 or dp[i - len(word)]:
                if s[i - len(word) + 1: i + 1] == word:
                    dp[i] = True
                    break
    return dp[-1]
