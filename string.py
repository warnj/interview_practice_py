# https://leetcode.com/problems/longest-substring-without-repeating-characters/
# two pointer: O(n) time and O(1) space (max set size limited by # of different chars)
import re
from collections import deque, Counter, defaultdict
from functools import cache
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
# above answer requires 2n steps in worst case, this optimization requires n at most
def lengthOfLongestSubstringOptimal(self, s: str) -> int:
    n = len(s)
    ans = 0
    charToNextIndex = {} # stores the index after current character
    i = 0
    # try to extend the range [i, j]
    for j in range(n):
        if s[j] in charToNextIndex: # the char s[j] has already been seen
            # either leave i (low pointer) where it is if previously seen location was before the start of current
            # window or move the low end of window to the next location after the duplicate char
            i = max(charToNextIndex[s[j]], i)

        ans = max(ans, j - i + 1)
        charToNextIndex[s[j]] = j + 1
    return ans

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

# https://leetcode.com/problems/palindromic-substrings
# O(n^2) time and O(1) space
def countSubstrings(self, s: str) -> int:
    def palindromeCountOdd(s, i):
        count = 1
        lo = i - 1
        hi = i + 1
        while lo >= 0 and hi < len(s) and s[lo] == s[hi]:
            count += 1
            lo -= 1
            hi += 1
        return count
    def palindromeCountEven(s, lo, hi):
        count = 0
        while lo >= 0 and hi < len(s) and s[lo] == s[hi]:
            count += 1
            lo -= 1
            hi += 1
        return count
    result = 1
    for i in range(1, len(s)):  # check possible centers
        result += palindromeCountOdd(s, i) + palindromeCountEven(s, i - 1, i)
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
# n=len(s), m=len(worddict), k=avg(len(words))  Time complexity: O(n⋅m⋅k) Space complexity: O(n)
def wordBreakMemo(self, s: str, wordDict: List[str]) -> bool:
    @cache
    def dp(i):
        if i < 0:
            return True
        for word in wordDict:
            if s[i - len(word) + 1: i + 1] == word and dp(i - len(word)):
                return True
        return False
    return dp(len(s) - 1)
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

# https://leetcode.com/problems/remove-invalid-parentheses
# O(2^n) time and O(n) space (for recursion stack for each index)
def removeInvalidParentheses(self, s: str) -> List[str]:
    # find the min of each type to remove, try all combinations of removing them, return the valid ones
    openRemove, closeRemove = 0, 0
    opens = 0
    for c in s:
        if c == '(':
            opens += 1
        elif c == ')':
            if opens == 0:
                closeRemove += 1
            else:
                opens -= 1
    openRemove = opens

    result = set()
    def removeParens(strList, unclosedCount, i, oRemove, cRemove):
        if oRemove < 0 or cRemove < 0 or unclosedCount < 0:
            return  # prune this path in recursion tree, it will always be invalid
        if oRemove == 0 and cRemove == 0 and unclosedCount == 0 and i == len(strList):
            result.add(''.join(strList))
        elif i == len(strList):
            return
        elif strList[i] == '(':
            if oRemove > 0:
                # explore removing the open
                strList[i] = ''
                removeParens(strList, unclosedCount, i+1, oRemove-1, cRemove)
                strList[i] = '('
            # explore keeping it
            removeParens(strList, unclosedCount+1, i+1, oRemove, cRemove)
        elif strList[i] == ')':
            if cRemove > 0:
                # explore removing the close
                strList[i] = ''
                removeParens(strList, unclosedCount, i+1, oRemove, cRemove-1)
                strList[i] = ')'
            # explore keeping it
            removeParens(strList, unclosedCount-1, i+1, oRemove, cRemove)
        else:
            # explore keeping it
            removeParens(strList, unclosedCount, i+1, oRemove, cRemove)

    removeParens(list(s), 0, 0, openRemove, closeRemove)
    return list(result)

# https://leetcode.com/problems/multiply-strings
def multiply(self, num1: str, num2: str) -> str:
    # manual multiplication as you'd do by hand
    sums = []
    jNum = 0
    for j in range(len(num2) - 1, -1, -1):
        carry = 0
        temp = 0
        iNum = 0
        for i in range(len(num1) - 1, -1, -1):
            num = int(num1[i]) * int(num2[j]) + carry
            carry = num // 10
            temp += num % 10 * 10 ** iNum  # form number right to left
            iNum += 1
        if carry:
            temp += carry * 10 ** iNum
        sums.append(temp * 10 ** jNum)  # add some zeros to the right side for subsequent nums to sum
        jNum += 1
    return str(sum(sums))

# https://leetcode.com/problems/word-break-ii
# O(n * 2^n) time and O(2^n) space
def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
    wordDict = set(wordDict)
    result = []

    def wordBreakHelper(curWords, i):
        if i >= len(s):  # found a combination of words
            result.append(' '.join(curWords))
        else:
            for j in range(i + 1, len(s) + 1):
                if s[i:j] in wordDict:
                    # explore later options with the found word
                    curWords.append(s[i:j])
                    wordBreakHelper(curWords, j)
                    curWords.pop()

    wordBreakHelper([], 0)
    return result

# https://leetcode.com/problems/letter-combinations-of-a-phone-number
def letterCombinationsIterative(self, digits: str) -> List[str]:
    dToL = {
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz'
    }
    oldCombinations = [""]
    for d in digits:
        newCombinations = []
        for c in oldCombinations:
            for l in dToL[d]:
                newCombinations.append(c + l)
        oldCombinations = newCombinations
    return [] if len(oldCombinations) == 1 else oldCombinations
def letterCombinations(self, digits: str) -> List[str]:
    dToL = {
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz'
    }
    def explore(result, temp, digits, i):
        if i == len(digits):
            result.append(''.join(temp))
        else:
            for l in dToL[digits[i]]:
                temp.append(l)
                explore(result, temp, digits, i + 1)
                temp.pop()
    result = []
    if digits:
        explore(result, [], digits, 0)
    return result

# https://leetcode.com/problems/add-strings
def addStrings(self, num1: str, num2: str) -> str:
    i = len(num1)-1
    j = len(num2)-1
    result = []

    carry = 0
    while i >= 0 or j >= 0:
        n1 = 0 if i < 0 else int(num1[i])
        n2 = 0 if j < 0 else int(num2[j])
        digitSum = n1 + n2 + carry
        if digitSum > 9:
            carry = 1
            digitSum %= 10
        else:
            carry = 0
        result.append(str(digitSum))
        i -= 1
        j -= 1
    if carry:
        result.append(str(carry))
    return ''.join(reversed(result))

# https://leetcode.com/problems/longest-common-prefix
def longestCommonPrefix(self, strs: List[str]) -> str:
    pre = []
    i = 0
    while True:
        if i == len(strs[0]):
            return "".join(pre)
        c = strs[0][i]
        for j in range(1, len(strs)):
            s = strs[j]
            if i == len(s):
                return "".join(pre)
            if s[i] != c:
                return "".join(pre)
        pre.append(c)
        i += 1
    return "".join(pre)

# https://leetcode.com/problems/longest-common-subsequence
# O(n*m) time and space
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    l1 = len(text1)
    l2 = len(text2)
    dp = [[0] * l1 for _ in range(l2)]
    for i in range(l2): # col
        for j in range(l1): # row
            if text1[j] == text2[i]:
                # must use previous row and col to avoid double counting subsequent matching chars in same row/col
                dp[i][j] = 1 if i-i < 0 or j-1 < 0 else dp[i-1][j-1] + 1
            else:
                up = 0 if j-1 < 0 else dp[i][j-1]
                left = 0 if i-1 < 0 else dp[i-1][j]
                dp[i][j] = max(up, left)
    # for row in dp:
    #     print(row)
    return dp[l2-1][l1-1]
#   a b c d e
# a 1 1 1 1 1
# c 1 1 2 2 2
# e 1 1 2 2 3

#   a b c b a
# a 1 1 1 1 1
# b 1 2 2 2 2
# c 1 2 3 3 3
# b 1 2 3 4 4
# c 1 2 3 4 4
# a 1 2 3 4 5

#   a a a a
# a 1 1 1 1

# https://leetcode.com/problems/minimum-window-substring
# O(n+m) time and O(26*2) space
def minWindow(self, s: str, t: str) -> str:
    counts = Counter(t)  # char counts in smaller string t
    windowCounts = {}  # char counts in current window (lo and hi inclusive)
    numCountsGtOrEq = 0  # number of counts in current window that satisfy the count requirements in t
    lo = -1
    result = ''
    for hi in range(len(s)):
        c = s[hi]
        if c in counts:
            count = windowCounts.get(c, 0)
            windowCounts[c] = count + 1

            if numCountsGtOrEq < len(counts):  # saving the start and end of first legal sequence
                if lo == -1:
                    lo = hi  # save first as start
                if counts[c] == count + 1:
                    numCountsGtOrEq += 1  # window now meeting the count requirement for this character
                    if numCountsGtOrEq == len(counts):
                        result = s[lo: hi + 1]  # completing first legal sequence

            if numCountsGtOrEq == len(counts):
                # already found a legal sequence, can it be shorter? - shorten from the front if possible
                while lo < hi:
                    c2 = s[lo]
                    if c2 in counts:
                        count = counts[c2]
                        wCount = windowCounts[c2]
                        if wCount > count:
                            windowCounts[c2] -= 1  # lower the count and keep shrinking window
                        else:  # wCount == count
                            break  # keep the lo pointer here since window needs this char
                    lo += 1

        if hi - lo + 1 < len(result):
            result = s[lo: hi + 1]
    return result

# https://leetcode.com/problems/reverse-words-in-a-string
def reverseWords(self, s: str) -> str:
    l = list(s.strip())
    self.reverse(l, 0, len(l) - 1)
    lo = 0
    for i in range(len(l)):
        c = l[i]
        if c != ' ':
            if i > 0 and (l[i - 1] == ' ' or l[i - 1] == ''):
                # start of a word
                lo = i
            elif (i + 1 < len(l) and l[i + 1] == ' ') or i + 1 == len(l):
                # end of a word
                hi = i
                self.reverse(l, lo, hi)
        else:
            j = i + 1
            while j < len(l) and l[j] == ' ':
                l[j] = ''  # delete duplicate non-letter chars
                j += 1
            i = j
    return ''.join(l)
def reverse(self, letters, lo, hi):
    while lo < hi:
        letters[lo], letters[hi] = letters[hi], letters[lo]
        lo += 1
        hi -= 1
def reverseWordsSimple(self, s: str) -> str:
    words = s.split()
    words.reverse()
    return ' '.join(words)

# https://leetcode.com/problems/minimum-number-of-steps-to-make-two-strings-anagram
def minSteps(self, s: str, t: str) -> int:
    sCounts = [0] * 26
    for c in s:
        sCounts[ord(c) - ord('a')] += 1
    tCounts = [0] * 26
    for c in t:
        tCounts[ord(c) - ord('a')] += 1
    steps = 0
    for i in range(26):
        sc = sCounts[i]
        tc = tCounts[i]
        steps += abs(tc - sc)
    return steps // 2

# https://leetcode.com/problems/longest-duplicate-substring
# O(n logn) time and O(n) space
def longestDupSubstring(self, s: str) -> str:
    def duplicateSubstr(l):
        lo = 0
        hi = l
        subs = set()
        while hi <= len(s):
            sub = s[lo:hi]
            if sub in subs:
                return sub
            subs.add(sub)
            hi += 1
            lo += 1
        return ''
    r = ''
    hi = len(s) - 1
    lo = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        d = duplicateSubstr(mid)
        if len(d) > len(r): r = d
        if d:
            lo = mid + 1
        else:
            hi = mid - 1
    return r
# O(n^2) time and space
def longestDupSubstringBrute(self, s: str) -> str:
    counts = defaultdict(int)
    for i in range(len(s)):
        for j in range(i, len(s) + 1):
            substr = s[i:j]
            counts[substr] += 1
    r = ''
    for substr, count in counts.items():
        if count > 1 and len(substr) > len(r):
            r = substr
    return r

# https://leetcode.com/problems/clear-digits
def clearDigits(self, s: str) -> str:
    answer = []
    for char in s:
        if char.isdigit() and answer:
            answer.pop()
        else:
            answer.append(char) # takes advantage of the guarantee that all digits can be deleted from input
    return "".join(answer)
def clearDigitsNaive(self, s: str) -> str:
    s = list(s)
    for i in range(len(s)):
        if s[i].isdigit():
            lo = i - 1
            while lo >= 0:
                if s[lo].isalpha():
                    # delete them
                    s[i] = ''
                    s[lo] = ''
                    break
                lo -= 1
    return ''.join(s)

# https://leetcode.com/problems/goat-latin
def toGoatLatin(self, sentence: str) -> str:
    parts = sentence.split()
    for i in range(len(parts)):
        if not parts[i].lower().startswith(('a', 'e', 'i', 'o', 'u')):
            front = parts[i][0]
            parts[i] = parts[i][1:] + front
        parts[i] += 'ma'
        parts[i] += 'a' * (1 + i)
    return ' '.join(parts)

# https://leetcode.com/problems/merge-strings-alternately
def mergeAlternately(self, word1: str, word2: str) -> str:
    # word1 = word1.split()
    # word2 = word2.split()
    r = []
    for a, b in zip(word1, word2):
        r.append(a + b)
    l1 = len(word1)
    l2 = len(word2)
    if l1 > l2:
        r.extend(word1[l2:])
    else:
        r.extend(word2[l1:])
    return ''.join(r)
def mergeAlternatelyPretty(self, word1, word2):
    result = []
    n = max(len(word1), len(word2))
    for i in range(n):
        if i < len(word1):
            result += word1[i]
        if i < len(word2):
            result += word2[i]
    return "".join(result)

# https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string
def removeDuplicates(self, s: str) -> str:
    stack = []
    for i in range(len(s)):
        if stack and stack[-1] == s[i]:
            stack.pop()
        else:
            stack.append(s[i])
    return ''.join(stack)
# O(n^2) time dumb soln
def removeDuplicatesBrute(self, s: str) -> str:
    s = list(s)
    for i in range(1, len(s)):
        lo = i - 1
        hi = i
        while lo >= 0 and hi < len(s):
            while lo >= 0 and s[lo] == '': lo -= 1
            while hi < len(s) and s[hi] == '': hi += 1
            if lo >= 0 and hi < len(s) and s[lo] == s[hi]:
                s[lo] = ''
                s[hi] = ''
                lo -= 1
                hi += 1
            else:
                break
    return ''.join(s)

# https://leetcode.com/problems/custom-sort-string
def customSortString(self, order: str, s: str) -> str:
    r = []
    counts = [0] * 26  # count of chars in s
    for c in s:
        counts[ord(c) - ord('a')] += 1
    for o in order:
        count = counts[ord(o) - ord('a')]
        r.extend(o * count)
        counts[ord(o) - ord('a')] = 0
    for i, count in enumerate(counts):
        r.extend(chr(ord('a') + i) * count)
    return ''.join(r)

# https://leetcode.com/problems/check-if-one-string-swap-can-make-strings-equal
def areAlmostEqual(self, s1: str, s2: str) -> bool:
    if s1 == s2: return True
    lo, hi = -1, -1  # should be exactly 2 places they don't match
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            if lo < 0:
                lo = i
            elif hi < 0:
                hi = i
            else:
                return False
    return s1[lo] == s2[hi] and s1[hi] == s2[lo]

# https://leetcode.com/problems/word-ladder
# Time and space Complexity: O(M^2 * n)
def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    if endWord not in wordList: return 0

    def adjacent(w1, w2):
        diff = False
        for i, j in zip(w1, w2):
            if i != j:
                if diff:
                    return False
                diff = True
        return True

    # make graph of words 1 away from each other
    graph = defaultdict(list)
    combined = [beginWord] + wordList
    for i in range(len(combined) - 1):
        for j in range(i + 1, len(combined)):
            w1, w2 = combined[i], combined[j]
            if adjacent(w1, w2):
                graph[w1].append(w2)
                graph[w2].append(w1)
    # print(graph)
    # bfs
    q = deque([beginWord])
    visited = set([beginWord])
    d = 0
    while q:
        d += 1
        size = len(q)
        for _ in range(size):
            cur = q.popleft()
            # print('visiting', cur, 'at distance', d)
            if cur == endWord:
                return d
            for child in graph[cur]:
                if child not in visited:
                    visited.add(child)
                    q.append(child)
    return 0

# https://leetcode.com/problems/text-justification
# O(n*k) time
def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
    def get_words(i):
        current_line = []
        curr_length = 0
        while i < len(words) and curr_length + len(words[i]) <= maxWidth:
            current_line.append(words[i])
            curr_length += len(words[i]) + 1
            i += 1
        return current_line

    def create_line(line, i):
        base_length = -1
        for word in line:
            base_length += len(word) + 1

        extra_spaces = maxWidth - base_length
        if len(line) == 1 or i == len(words):
            return " ".join(line) + " " * extra_spaces

        word_count = len(line) - 1
        spaces_per_word = extra_spaces // word_count
        needs_extra_space = extra_spaces % word_count

        for j in range(needs_extra_space):
            line[j] += " "
        for j in range(word_count):
            line[j] += " " * spaces_per_word
        return " ".join(line)

    ans = []
    i = 0 # word index we're currently on
    while i < len(words):
        current_line = get_words(i)
        i += len(current_line)
        ans.append(create_line(current_line, i))
    return ans

# https://leetcode.com/problems/decode-string/
# Time Complexity: O(maxK⋅n), Space Complexity: O(m+n)
'''
class Solution {
    String decodeString(String s) {
        Stack<Integer> countStack = new Stack<>();
        Stack<StringBuilder> stringStack = new Stack<>();
        StringBuilder currentString = new StringBuilder();
        int k = 0;
        for (char ch : s.toCharArray()) {
            if (Character.isDigit(ch)) {
                k = k * 10 + ch - '0';
            } else if (ch == '[') {
                // push the number k to countStack
                countStack.push(k);
                // push the currentString to stringStack
                stringStack.push(currentString);
                // reset currentString and k
                currentString = new StringBuilder();
                k = 0;
            } else if (ch == ']') {
                StringBuilder decodedString = stringStack.pop();
                // decode currentK[currentString] by appending currentString k times
                for (int currentK = countStack.pop(); currentK > 0; currentK--) {
                    decodedString.append(currentString);
                }
                currentString = decodedString;
            } else {
                currentString.append(ch);
            }
        }
        return currentString.toString();
    }
}
# Time Complexity: O(maxK⋅n), O(n) space
class Solution {
    int index = 0;
    String decodeString(String s) {
        StringBuilder result = new StringBuilder();
        while (index < s.length() && s.charAt(index) != ']') {
            if (!Character.isDigit(s.charAt(index)))
                result.append(s.charAt(index++));
            else {
                int k = 0;
                // build k while next character is a digit
                while (index < s.length() && Character.isDigit(s.charAt(index)))
                    k = k * 10 + s.charAt(index++) - '0';
                // ignore the opening bracket '['    
                index++;
                String decodedString = decodeString(s);
                // ignore the closing bracket ']'
                index++;
                // build k[decodedString] and append to the result
                while (k-- > 0)
                    result.append(decodedString);
            }
        }
        return new String(result);
    }
}
'''

# https://leetcode.com/problems/regular-expression-matching
def isMatch(self, text: str, pattern: str) -> bool:
    if not pattern:
        return not text
    first_match = bool(text) and pattern[0] in {text[0], "."}
    if len(pattern) >= 2 and pattern[1] == "*":
        return (
            self.isMatch(text, pattern[2:])
            or first_match
            and self.isMatch(text[1:], pattern)
        )
    else:
        return first_match and self.isMatch(text[1:], pattern[1:])
def isMatch(self, text: str, pattern: str) -> bool:
    memo = {}
    def dp(i: int, j: int) -> bool:
        if (i, j) not in memo:
            if j == len(pattern):
                ans = i == len(text)
            else:
                first_match = i < len(text) and pattern[j] in {text[i], "."}
                if j + 1 < len(pattern) and pattern[j + 1] == "*":
                    ans = dp(i, j + 2) or first_match and dp(i + 1, j)
                else:
                    ans = first_match and dp(i + 1, j + 1)

            memo[i, j] = ans
        return memo[i, j]
    return dp(0, 0)
# time and space complexity is O(TP)
def isMatch(self, text: str, pattern: str) -> bool:
    dp = [[False] * (len(pattern) + 1) for _ in range(len(text) + 1)]
    dp[-1][-1] = True
    for i in range(len(text), -1, -1):
        for j in range(len(pattern) - 1, -1, -1):
            first_match = i < len(text) and pattern[j] in {text[i], "."}
            if j + 1 < len(pattern) and pattern[j + 1] == "*":
                dp[i][j] = dp[i][j + 2] or first_match and dp[i + 1][j]
            else:
                dp[i][j] = first_match and dp[i + 1][j + 1]
    return dp[0][0]

# https://leetcode.com/problems/distinct-subsequences/
# time and space O(n*m)
def numDistinct(self, s: str, t: str) -> int:
    memo = {}
    def uniqueSubsequences(i: int, j: int) -> int:
        M, N = len(s), len(t)
        # Base case
        if i == M or j == N or M - i < N - j:
            return int(j == len(t))
        # Check if the result is already cached
        if (i, j) in memo:
            return memo[i, j]
        # Always make this recursive call
        ans = uniqueSubsequences(i + 1, j)
        # If the characters match, make the other
        # one and add the result to "ans"
        if s[i] == t[j]:
            ans += uniqueSubsequences(i + 1, j + 1)
        # Cache the answer and return
        memo[i, j] = ans
        return ans
    return uniqueSubsequences(0, 0)
# time and space O(n*m)
def numDistinct(self, s: str, t: str) -> int:
    M: int = len(s)
    N: int = len(t)
    # Dynamic Programming table
    dp: List[List[int]] = [[0 for _ in range(N + 1)] for _ in range(M + 1)]

    # Base case initialization
    for j in range(N + 1):
        dp[M][j] = 0
    # Base case initialization
    for i in range(M + 1):
        dp[i][N] = 1

    # Iterate over the strings in reverse so as to
    # satisfy the way we've modeled our recursive solution
    for i in range(M - 1, -1, -1):
        for j in range(N - 1, -1, -1):
            # Remember, we always need this result
            dp[i][j] = dp[i + 1][j]
            # If the characters match, we add the
            # result of the next recursion call (in this
            # case, the value of a cell in the dp table)
            if s[i] == t[j]:
                dp[i][j] += dp[i + 1][j + 1]
    return dp[0][0]
# time O(n*m), space=O(n)
def numDistinct(self, s: str, t: str) -> int:
    M, N = len(s), len(t)
    # Dynamic Programming table
    dp = [0 for j in range(N)]
    # Iterate over the strings in reverse so as to
    # satisfy the way we've modeled our recursive solution
    for i in range(M - 1, -1, -1):
        # At each step we start with the last value in
        # the row which is always 1. Notice how we are
        # starting the loop from N - 1 instead of N like
        # in the previous solution.
        prev = 1
        for j in range(N - 1, -1, -1):
            # Record the current value in this cell so that
            # we can use it to calculate the value of dp[j - 1]
            old_dpj = dp[j]
            # If the characters match, we add the
            # result of the next recursion call (in this
            # case, the value of a cell in the dp table
            if s[i] == t[j]:
                dp[j] += prev
            # Update the prev variable
            prev = old_dpj
    return dp[0]

# https://leetcode.com/problems/expression-add-operators
# time O(n*4^n), space O(n)
def addOperators(self, num: 'str', target: 'int') -> 'List[str]':
    N = len(num)
    answers = []
    def recurse(index, prev_operand, current_operand, value, string):

        # Done processing all the digits in num
        if index == N:

            # If the final value == target expected AND
            # no operand is left unprocessed
            if value == target and current_operand == 0:
                answers.append("".join(string[1:]))
            return

        # Extending the current operand by one digit
        current_operand = current_operand*10 + int(num[index])
        str_op = str(current_operand)

        # To avoid cases where we have 1 + 05 or 1 * 05 since 05 won't be a
        # valid operand. Hence this check
        if current_operand > 0:

            # NO OP recursion
            recurse(index + 1, prev_operand, current_operand, value, string)

        # ADDITION
        string.append('+'); string.append(str_op)
        recurse(index + 1, current_operand, 0, value + current_operand, string)
        string.pop();string.pop()

        # Can subtract or multiply only if there are some previous operands
        if string:

            # SUBTRACTION
            string.append('-'); string.append(str_op)
            recurse(index + 1, -current_operand, 0, value - current_operand, string)
            string.pop();string.pop()

            # MULTIPLICATION
            string.append('*'); string.append(str_op)
            recurse(index + 1, current_operand * prev_operand, 0, value - prev_operand + (current_operand * prev_operand), string)
            string.pop();string.pop()
    recurse(0, 0, 0, 0, [])
    return answers
