import heapq
from bisect import bisect_left, bisect_right
from collections import deque, Counter, defaultdict
import random
from typing import List, Optional

from sortedcontainers import SortedList

from premium import NestedInteger


# https://leetcode.com/problems/find-score-of-an-array-after-marking-all-elements
# O(nlogn) time and O(n) space
def findScore(self, nums: List[int]) -> int:
    marked = {}  # index in list -> true - indicating marked
    tuples = [(n, i) for i, n in enumerate(nums)]
    tuples.sort()
    score = 0
    for t in tuples:
        if t[1] not in marked:
            score += t[0]
            marked[t[1]] = True
            marked[t[1] + 1] = True
            marked[t[1] - 1] = True
    return score


# https://leetcode.com/problems/continuous-subarrays
def continuousSubarraysBrute(self, nums: List[int]) -> int:
    # order matters, sorting would mess the result up
    # brute force:
    # find all possible subarrays
    # of these, if the difference between max and min is >2, don't count it
    # a lot of duplicate work in finding maxes and mins
    # runtime: O(n^3), space: O(n^2)
    allSubarrays = _all_subarrays_over_2(nums)
    count = len(nums)
    for s in allSubarrays:
        if max(s) - min(s) <= 2:
            count += 1
    return count
# returns all subarrays of the given array with a min length of 2
def _all_subarrays_over_2(array):
    subarrays = []
    for i in range(len(array)):
        for j in range(i+2, len(array)+1):
            subarrays.append(array[i:j])
    return subarrays
# O(nlogk)≈O(n) time, O(k)≈O(1) space, (k bounded by 3)
def continuousSubarraysSortedMap(self, nums: List[int]) -> int:
    # Map to maintain sorted frequency map of current window
    freq = {}
    left = right = 0
    count = 0  # Total count of valid subarrays
    while right < len(nums):
        # Add current element to frequency map
        freq[nums[right]] = freq.get(nums[right], 0) + 1
        # While window violates the condition |nums[i] - nums[j]| ≤ 2
        # Shrink window from left
        while max(freq) - min(freq) > 2:
            # Remove leftmost element from frequency map
            freq[nums[left]] -= 1
            if freq[nums[left]] == 0:
                del freq[nums[left]]
            left += 1
        # Add count of all valid subarrays ending at right
        count += right - left + 1
        right += 1
    return count
# O(nlogn) time and O(n) space
def continuousSubarraysHeap(self, nums: List[int]) -> int:
    # Two heaps to track min/max indices, sorted by nums[index]
    min_heap = []  # (nums[i], i) tuples for min tracking
    max_heap = []  # (-nums[i], i) tuples for max tracking
    left = right = 0
    count = 0
    while right < len(nums):
        # Add current index to both heaps
        # For max heap, negate value to convert min heap to max heap
        heapq.heappush(min_heap, (nums[right], right))
        heapq.heappush(max_heap, (-nums[right], right))
        # While window violates |nums[i] - nums[j]| ≤ 2
        # Shrink window from left and remove outdated indices
        while left < right and -max_heap[0][0] - min_heap[0][0] > 2:
            left += 1
            # Remove indices outside window from both heaps
            while min_heap and min_heap[0][1] < left:
                heapq.heappop(min_heap)
            while max_heap and max_heap[0][1] < left:
                heapq.heappop(max_heap)
        count += right - left + 1
        right += 1
    return count
def continuousSubarraysDeque(self, nums: List[int]) -> int:
    # Monotonic deque to track maximum and minimum elements
    max_q = deque()
    min_q = deque()
    left = 0
    count = 0
    for right, num in enumerate(nums):
        # Maintain decreasing monotonic deque for maximum values
        while max_q and nums[max_q[-1]] < num:
            max_q.pop()
        max_q.append(right)
        # Maintain increasing monotonic deque for minimum values
        while min_q and nums[min_q[-1]] > num:
            min_q.pop()
        min_q.append(right)
        # Shrink window if max-min difference exceeds 2
        while max_q and min_q and nums[max_q[0]] - nums[min_q[0]] > 2:
            # Move left pointer past the element that breaks the condition
            if max_q[0] < min_q[0]:
                left = max_q[0] + 1
                max_q.popleft()
            else:
                left = min_q[0] + 1
                min_q.popleft()
        # Add count of all valid subarrays ending at current right pointer
        count += right - left + 1
    return count
def continuousSubarraysMath(self, nums: List[int]) -> int:
    right = left = 0
    window_len = total = 0
    # Initialize window with first element
    cur_min = cur_max = nums[right]
    for right in range(len(nums)):
        # Update min and max for current window
        cur_min = min(cur_min, nums[right])
        cur_max = max(cur_max, nums[right])
        # If window condition breaks (diff > 2)
        if cur_max - cur_min > 2:
            # Add subarrays from previous valid window
            window_len = right - left
            total += window_len * (window_len + 1) // 2
            # Start new window at current position
            left = right
            cur_min = cur_max = nums[right]
            # Expand left boundary while maintaining condition
            while left > 0 and abs(nums[right] - nums[left - 1]) <= 2:
                left -= 1
                cur_min = min(cur_min, nums[left])
                cur_max = max(cur_max, nums[left])
            # Remove overcounted subarrays if left boundary expanded
            if left < right:
                window_len = right - left
                total -= window_len * (window_len + 1) // 2
    # Add subarrays from final window
    window_len = right - left + 1
    total += window_len * (window_len + 1) // 2
    return total


# https://leetcode.com/problems/maximum-average-pass-ratio
# O(k * n) time, O(1) space
def maxAverageRatioBrute(self, classes: List[List[int]], extraStudents: int) -> float:
    for i in range(0, extraStudents):
        # find the maximum change in pass rate if a passing student was added
        maxIncr = 0
        maxClass = None
        for c in classes:
            if maxIncr < (c[0]+1)/(c[1]+1) - c[0]/c[1]:
                maxIncr = (c[0]+1)/(c[1]+1) - c[0]/c[1]
                maxClass = c
        maxClass[0] += 1
        maxClass[1] += 1
    return sum(x[0] / x[1] for x in classes) / len(classes)
def maxAverageRatioOptimized(self, classes: List[List[int]], extraStudents: int) -> float:
    classes = sorted(classes, key=lambda c: -((c[0]+1)/(c[1]+1) - c[0]/c[1]))
    for i in range(0, extraStudents):
        maxIncr = 0
        maxClass = None
        for j in range(0, min(i+2, len(classes))):  # only check up to first few classes since we pre-sorted
            c = classes[j]
            if maxIncr < (c[0]+1)/(c[1]+1) - c[0]/c[1]:
                maxIncr = (c[0]+1)/(c[1]+1) - c[0]/c[1]
                maxClass = c
        maxClass[0] += 1
        maxClass[1] += 1
    return sum(x[0] / x[1] for x in classes) / len(classes)
# O(n + k*log(n)) time, O(n) space
def maxAverageRatioHeap(self, classes: List[List[int]], extraStudents: int) -> float:
    max_diffs = [(-((c[0]+1)/(c[1]+1) - c[0]/c[1]), c[0], c[1]) for c in classes]  # min heap of the negative diffs
    heapq.heapify(max_diffs)
    for k in range(0, extraStudents):
        _, c0, c1 = heapq.heappop(max_diffs)
        c0 += 1
        c1 += 1
        heapq.heappush(max_diffs, (-((c0+1)/(c1+1) - c0/c1), c0, c1))
    return sum(x[1] / x[2] for x in max_diffs) / len(classes)


# https://leetcode.com/problems/final-array-state-after-k-multiplication-operations-i
def getFinalState(self, nums: List[int], k: int, multiplier: int):
    pq = [(val, i) for i, val in enumerate(nums)]
    heapq.heapify(pq)
    for _ in range(k):
        _, i = heapq.heappop(pq)
        nums[i] *= multiplier
        heapq.heappush(pq, (nums[i], i))
    return nums


# https://leetcode.com/problems/construct-string-with-repeat-limit
# n = # chars in s:   O(n) time and O(1) space (considering 26 a constant)
def repeatLimitedString(self, s: str, repeatLimit: int) -> str:
    counts = {}
    for c in s:
        if c in counts:
            counts[c] += 1
        else:
            counts[c] = 1
    result = ''
    repeat = 0
    for char_code in range(ord('z'), ord('a') - 1, -1):
        while chr(char_code) in counts and counts[chr(char_code)] > 0:
            result += chr(char_code)
            counts[chr(char_code)] -= 1
            repeat += 1
            if repeat == repeatLimit and counts[chr(char_code)] > 0:
                # find next letter and place 1 to break up the repeat
                breakup = None
                for char_code2 in range(char_code-1, ord('a') - 1, -1):
                    if chr(char_code2) in counts and counts[chr(char_code2)] > 0:
                        breakup = chr(char_code2)
                        break
                if not breakup: # no other options to stop the repeat, end early
                    return result
                repeat = 0
                result += chr(char_code2)
                counts[chr(char_code2)] -= 1
        repeat = 0
    return result
# O(n*k) time (k=26) and O(k) space
def repeatLimitedStringEditorial(self, s: str, repeatLimit: int) -> str:
    freq = [0] * 26
    for char in s:
        freq[ord(char) - ord("a")] += 1
    result = []
    current_char_index = 25  # Start from the largest character
    while current_char_index >= 0:
        if freq[current_char_index] == 0:
            current_char_index -= 1
            continue
        use = min(freq[current_char_index], repeatLimit)
        result.append(chr(current_char_index + ord("a")) * use)
        freq[current_char_index] -= use
        if freq[current_char_index] > 0:  # Need to add a smaller character
            smaller_char_index = current_char_index - 1
            while smaller_char_index >= 0 and freq[smaller_char_index] == 0:
                smaller_char_index -= 1
            if smaller_char_index < 0:
                break
            result.append(chr(smaller_char_index + ord("a")))
            freq[smaller_char_index] -= 1
    return "".join(result)
# O(n*logk) time and O(k) space
def repeatLimitedStringHeap(self, s: str, repeatLimit: int) -> str:
    max_heap = [(-ord(c), cnt) for c, cnt in Counter(s).items()]
    heapq.heapify(max_heap)
    result = []
    while max_heap:
        char_neg, count = heapq.heappop(max_heap)
        char = chr(-char_neg)
        use = min(count, repeatLimit)
        result.append(char * use)
        if count > use and max_heap:
            next_char_neg, next_count = heapq.heappop(max_heap)
            result.append(chr(-next_char_neg))
            if next_count > 1:
                heapq.heappush(max_heap, (next_char_neg, next_count - 1))
            heapq.heappush(max_heap, (char_neg, count - use))
    return "".join(result)


# https://leetcode.com/problems/final-prices-with-a-special-discount-in-a-shop
# O(n^2) time, O(1) space
def finalPricesBrute(self, prices: List[int]) -> List[int]:
    for i in range(len(prices)):
        discount = 0
        for j in range(i+1, len(prices)):
            if prices[j] <= prices[i]:
                discount = prices[j]
                break
        prices[i] -= discount
    return prices
def finalPrices(self, prices: List[int]) -> List[int]:
    # Create a copy of prices array to store discounted prices
    result = prices.copy()
    stack = deque()
    for i in range(len(prices)):
        # Process items that can be discounted by current price
        while stack and prices[stack[-1]] >= prices[i]:
            # Apply discount to previous item using current price
            result[stack.pop()] -= prices[i]
        # Add current index to stack
        stack.append(i)
    return result


# https://leetcode.com/problems/max-chunks-to-make-sorted
def maxChunksToSorted(self, arr: List[int]) -> int:
    n = len(arr)
    chunks = 0
    max_element = 0
    # Iterate over the array
    for i in range(n):
        # Update max_element
        max_element = max(max_element, arr[i])
        if max_element == i:
            # All values in range [0, i] belong to the prefix arr[0:i]; a chunk can be formed
            chunks += 1
    return chunks


# https://leetcode.com/problems/merge-intervals
# O(nlogn) time O(n) space (python sort uses Timsort worstcase O(n))
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    sorted_lists = sorted(intervals, key=lambda x: x[0])
    result = []
    lo = sorted_lists[0][0]
    hi = sorted_lists[0][1]
    for i in range(len(sorted_lists)):
        l = sorted_lists[i]
        if l[0] > hi:
            # end current interval and start a new one
            result.append([lo, hi])
            lo = l[0]
            hi = l[1]
        else:
            # expand current interval
            hi = max(hi, l[1])

    result.append([lo, hi])
    return result

# https://leetcode.com/problems/insert-interval
# O(n) time and O(1) space
def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    # todo: binary search for starting point
    i = 0
    while i < len(intervals) and intervals[i][0] < newInterval[0]:
        i += 1

    # i is where the newInterval should go, now insert in middle and possibly merge
    result = []
    cur = None
    if i - 1 >= 0 and intervals[i - 1][1] >= newInterval[0]:
        # need to merge with the one interval before
        result.extend(intervals[:i - 1])
        cur = [intervals[i - 1][0], max(intervals[i - 1][1], newInterval[1])]
    else:
        result.extend(intervals[:i])  # add the non-overlapping beginning intervals
        cur = newInterval

    # build new combined interval; merge from cur to the end of intervals adding to new result
    while i < len(intervals) and cur[1] >= intervals[i][0]:
        cur[1] = max(cur[1], intervals[i][1])
        i += 1
    result.append(cur)

    # add the non-overlapping ending intervals to result
    result.extend(intervals[i:])
    return result

# https://leetcode.com/problems/interval-list-intersections
def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
    def overlap(a, b):
        a, b = (a, b) if a[0] <= b[0] else (b, a) # a will have the lowest first value
        if a[1] >= b[0]:
            return [b[0], min(a[1], b[1])] # overlap is highest first val and lowest second val
        return None

    result = []
    i, j = 0, 0
    while i < len(firstList) and j < len(secondList):
        interval = overlap(firstList[i], secondList[j])
        if interval:
            result.append(interval)
        if firstList[i][1] > secondList[j][1]: # consider the interval that ends last for next overlap
            j += 1
        else:
            i += 1
    return result

# https://leetcode.com/problems/random-pick-with-weight/
# reduce original problem down to the problem of inserting an element into a sorted list
class RandomPickWeighted:
    # O(n) time and O(n) extra space
    def __init__(self, w: List[int]):
        self.sums = w[:]
        for i in range(1, len(w)):
            self.sums[i] += self.sums[i-1]  # i.e. {2, 5, 3, 4} => {2, 7, 10, 14}
    # O(logn) time and O(1) extra space
    def pickIndex(self) -> int:
        r = random.randint(1, self.sums[-1])
        # r in [1,2] return 0
        # r in [3,7] return 1
        # r in [8,10] return 2
        # r in [11,14] return 3
        lo = 0
        hi = len(self.sums)-1
        while lo < hi:
            mid = lo + (hi-lo) // 2
            prev = self.sums[mid-1] if mid-1 >= 0 else 0
            if prev < r <= self.sums[mid]:
                return mid
            if r > self.sums[mid]:
                lo = mid+1
            else:
                hi = mid-1
        return lo
class RandomPickWeightedBad:
    # [1,3]
    # 1 -> 0
    # 2 -> 1
    # 3 -> 1
    # 4 -> 1
    # O(n) time and O(1) extra space
    def __init__(self, w: List[int]):
        self.sum = sum(w)
        self.w = w
    # O(n) time and O(1) extra space
    def pickIndex(self) -> int:
        r = random.randint(1, self.sum)
        i = -1
        while r > 0:
            i += 1
            r -= self.w[i]
        return i

# https://leetcode.com/problems/powx-n/
# O(n) time and O(1) space
def myPowOriginal(self, x: float, n: int) -> float:
    # return x ** n
    r = 1
    for i in range(abs(n)):
        r *= x
    if n < 0:
        return 1 / r
    return r
# O(log n) time and space
def myPow(self, x: float, n: int) -> float:
    def function(base, exponent):
        if exponent == 0:
            return 1
        elif exponent % 2 == 0:
            return function(base * base, exponent // 2) # break the exponent in half with multiplication
        else:
            return base * function(base * base, (exponent - 1) // 2) # reduce the exponent by 1 with extra multiplication and then apply the double multiplication from even case
    f = function(x, abs(n))
    return f if n >= 0 else 1 / f

# https://leetcode.com/problems/reverse-integer
def reverse(self, x: int) -> int:
    neg = True if x < 0 else False
    if neg:
        x *= -1
    result = 0
    while x > 0:
        digit = x % 10
        x = x // 10
        result = result * 10 + digit

    if result > 2 ** 31 - 1 or result < -2 ** 31:
        return 0
    if neg:
        return -result
    return result

# https://leetcode.com/problems/climbing-stairs
# O(n) time and O(1) space
def climbStairs(self, n: int) -> int:
    if n <= 3:
        return n
    ways2 = 1
    ways1 = 2
    ways = 3
    for i in range(3, n):
        ways2, ways1 = ways1, ways
        ways = ways1 + ways2
    return ways

# https://leetcode.com/problems/pascals-triangle
def generate(self, numRows: int) -> List[List[int]]:
    result = [[1]]
    for i in range(2, numRows + 1):
        new = []
        for j in range(i):
            s = result[-1][j - 1] if j - 1 >= 0 else 0
            s += result[-1][j] if j < len(result[-1]) else 0
            new.append(s)
        result.append(new)
    return result
def generate2(self, numRows: int) -> List[List[int]]:
    result = [[1]]
    for i in range(2, numRows + 1):
        new = [1]
        prev = result[-1]
        for j in range(1, i - 1):
            new.append(prev[j - 1] + prev[j])
        new.append(1)
        result.append(new)
    return result

# https://leetcode.com/problems/first-bad-version
def isBadVersion(n):
    pass
def firstBadVersion(self, n: int) -> int:
    result = -1
    lo = 1
    hi = n
    while lo <= hi:
        mid = (lo + hi) // 2
        if isBadVersion(mid):  # first bad is to the left or we are on the solution
            hi = mid - 1
            result = mid
        else:
            lo = mid + 1
    return result

# https://leetcode.com/problems/count-and-say
# O(n*m) time and O(1) space
def countAndSay(self, n: int) -> str:
    def encodeRLE(numStr):
        result = []
        count = 1
        i = 1
        while i < len(numStr):
            if numStr[i] == numStr[i - 1]:
                count += 1
            else:
                result.append(str(count))
                result.append(numStr[i - 1])
                count = 1
            i += 1
        if i - 1 >= 0:
            result.append(str(count))
            result.append(numStr[i - 1])
        return ''.join(result)

    prevRLE = '1'
    for cur in range(2, n + 1):
        prevRLE = encodeRLE(prevRLE)
    return prevRLE
def countAndSayRecursive(self, n: int) -> str:
    def encodeRLE(numStr):
        result = []
        count = 1
        i = 1
        while i < len(numStr):
            if numStr[i] == numStr[i-1]:
                count += 1
            else:
                result.append(str(count))
                result.append(numStr[i-1])
                count = 1
            i += 1
        if i-1 >= 0:
            result.append(str(count))
            result.append(numStr[i-1])
        return ''.join(result)
    if n == 1:
        return '1'
    return encodeRLE(str(self.countAndSayRecursive(n-1)))

# https://leetcode.com/problems/string-to-integer-atoi
def myAtoiSoln(self, input: str) -> int:
    sign = 1
    result = 0
    index = 0
    n = len(input)
    INT_MAX = pow(2, 31) - 1
    INT_MIN = -pow(2, 31)
    while index < n and input[index] == " ":
        index += 1
    # sign = +1, if it's positive number, otherwise sign = -1.
    if index < n and input[index] == "+":
        sign = 1
        index += 1
    elif index < n and input[index] == "-":
        sign = -1
        index += 1
    # Traverse next digits of input and stop if it is not a digit. End of string is also non-digit character.
    while index < n and input[index].isdigit():
        digit = int(input[index])
        # Check overflow and underflow conditions.
        if (result > INT_MAX // 10) or (result == INT_MAX // 10 and digit > INT_MAX % 10):
            # If integer overflowed return 2^31-1, otherwise if underflowed return -2^31.
            return INT_MAX if sign == 1 else INT_MIN
        # Append current digit to the result.
        result = 10 * result + digit
        index += 1
    return sign * result
def myAtoi(self, s: str) -> int:
    def limitSize(s):
        # instead of doing "n = int(s)" you can do:
        neg = s[0] == '-'
        if neg:
            s = s[1:]
        n = 0
        for i in range(len(s)):
            n = n*10 + int(s[i])
        if neg:
            n *= -1

        if n > 2**31-1:
            return 2**31-1
        elif n < -2**31:
            return -2**31
        return n

    s = s.strip()
    if s:
        if s[0] == '+':
            if 1 < len(s) and not s[1].isnumeric():
                return 0
            s = s[1:]
        if s:
            start = 1 if s[0] == '-' else 0
            for i in range(start, len(s)):
                if not s[i].isnumeric():
                    if i > start:
                        return limitSize(s[:i])
                    else:
                        return 0
                if i == len(s)-1:
                    return limitSize(s)
    return 0

# https://leetcode.com/problems/palindrome-number
def isPalindrome(self, x: int) -> bool:
    if x < 0 or (x != 0 and x % 10 == 0):
        return False
    reversed_num = 0
    while x > reversed_num:
        reversed_num = reversed_num * 10 + x % 10
        x //= 10
    return x == reversed_num or x == reversed_num // 10
def isPalindrome(self, x: int) -> bool:
    reverse = 0
    temp = x
    while temp > 0:
        digit = temp % 10  # take last digit from temp
        temp = temp // 10

        reverse *= 10  # move previous digit to the left
        reverse += digit  # append to reversed number
    # temp = 123, 12, 1, 0
    # digit =     3, 2, 1
    # reverse = 0, 3, 32, 321
    return reverse == x

# https://leetcode.com/problems/maximum-swap
def maximumSwap(self, num: int) -> int:
    numStr = list(str(num))
    maxRightI = len(numStr) - 1
    swap1 = -1
    swap2 = -1

    for i in range(len(numStr) - 2, -1, -1):
        maxRightI = i + 1 if int(numStr[i + 1]) > int(numStr[maxRightI]) else maxRightI
        if int(numStr[maxRightI]) > int(numStr[i]):
            swap1 = maxRightI
            swap2 = i

    if swap1 > 0:
        numStr[swap1], numStr[swap2] = numStr[swap2], numStr[swap1]
        return int(''.join(numStr))
    return num
def maximumSwapBrute(self, num: int) -> int:
    numStr = str(num)
    result = num
    for i in range(len(numStr)-1):
        for j in range(1, len(numStr)):
            swapped = list(numStr)
            temp = swapped[j]
            swapped[j] = swapped[i]
            swapped[i] = temp
            result = max(result, int(''.join(swapped)))
    return result

# https://leetcode.com/problems/online-stock-span
# O(1) amortized lookup; while loop runs n times total, each of n elements gets popped a max of once
# monotonic decreasing stack: O(n) space
class StockSpannerPretty:
    def __init__(self):
        self.stack = []
    def next(self, price: int) -> int:
        ans = 1
        while self.stack and self.stack[-1][0] <= price:
            ans += self.stack.pop()[1]
        self.stack.append((price, ans))
        return ans
class StockSpanner:
    def __init__(self):
        self.stack = []  # decreasing stack of (price, answer)
        self.prices = []
        # goal:  find how far away first day with price > current is

        # Stack contents:
        #                          [(100,1), (80,1), (60,1)]
        #                          [(100,1), (80,1), (70,2)]
        #                          [(100,1), (80,1), (70,2), (60,1)]
        #                          [(100,1), (80,1), (75,4)]
        #                          [(100,1), (85,6)]
        # Example:
        #   [100,80,60,70,60,75,85]
        #   [1,  1, 1,  2, 1, 4, 6]
    def next(self, price: int) -> int:
        self.prices.append(price)
        i = len(self.prices) - 2
        while i >= 0:
            prev = self.prices[i]
            if prev > price:
                break
            else:
                # possible to use stack to avoid re-checking previous prices
                top = self.stack[-1]
                if top[0] <= price:
                    self.stack.pop() # remove this value to preserve the decreasing stack and check the prev value on next loop itr
                    i -= top[1]
                else:
                    i -= 1

        result = len(self.prices) - i - 1  # i is currently at the index of the most recent higher value than price (we want len of window excluding i, so subtract 1)
        self.stack.append((price, result)) # stack will always end with current price
        return result
class StockSpannerSimple: # O(n) time and space
    def __init__(self):
        self.past = []

    def next(self, price: int) -> int:
        result = 1
        self.past.append(price)
        for i in range(len(self.past)-2, -1, -1):
            if self.past[i] <= price:
                result += 1
            else:
                return result
        return result

# https://leetcode.com/problems/can-i-win

# https://leetcode.com/problems/find-duplicate-file-in-system/description/?envType=company&envId=netflix&favoriteSlug=netflix-all
# O(n*x) time and space where n is number of strings and x is avg string length
def findDuplicate(self, paths: List[str]) -> List[List[str]]:
    # map: content -> [file paths with this content]
    content = defaultdict(list)

    for path in paths:
        parts = path.split()
        d = parts[0]
        for i in range(1, len(parts)):
            # separate the filename and content
            nameContent = parts[i].split('(')
            n = nameContent[0]
            c = nameContent[1][:-1]
            content[c].append(d + '/' + n)
    result = []
    for c, p in content.items():
        if len(p) > 1:
            result.append(p)
    return result

# https://leetcode.com/problems/design-hit-counter
# Design a hit counter which counts the number of hits received in the past 5 minutes (i.e., the past 300 seconds).
# Your system should accept a timestamp parameter (in seconds granularity), and you may assume that calls are being made to the system in chronological order (i.e., timestamp is monotonically increasing). Several hits may arrive roughly at the same time.
# Implement the HitCounter class:
# HitCounter() Initializes the object of the hit counter system.
# void hit(int timestamp) Records a hit that happened at timestamp (in seconds). Several hits may happen at the same timestamp.
# int getHits(int timestamp) Returns the number of hits in the past 5 minutes from timestamp (i.e., the past 300 seconds).
class HitCounter:
    def __init__(self):
        self.hits = [] # store all hits ever
    def hit(self, timestamp: int) -> None:
        self.hits.append(timestamp)
    def getHits(self, timestamp: int) -> int:
        # binary search the start and end index of the sorted hit list
        left = bisect_left(self.hits, timestamp-299) # t-299 is inclusive lower bound of window
        right = bisect_right(self.hits, timestamp)
        return right-left
class HitCounter2:
    def __init__(self):
        self.hits = deque() # getHits will only be called with increasing params >= last hit
    def hit(self, timestamp: int) -> None:
        self.hits.append(timestamp)
    def getHits(self, timestamp: int) -> int:
        while self.hits and self.hits[0] < timestamp-299:
            self.hits.popleft()
        return len(self.hits)

# https://leetcode.com/problems/sliding-window-maximum
# O(n*logk) time and O(n) space
def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
    # max heap, add first k to it, store (val,index) tuples, move the window right adding new element, while top of heap is no longer in the window pop
    heap = [(-nums[i], i) for i in range(k)]
    heapq.heapify(heap)
    result = []
    lo = 0
    hi = k - 1
    while hi < len(nums):
        maximum, index = heap[0]
        result.append(-maximum)
        lo += 1
        hi += 1
        if hi == len(nums): break
        heapq.heappush(heap, (-nums[hi], hi))
        while heap[0][1] < lo:  # remove from the heap while it's outside the window
            heapq.heappop(heap)
    return result
# O(n) time and space
def maxSlidingWindowBest(self, nums: List[int], k: int) -> List[int]:
    # monotonic decreasing queue of the max in the current window and decreasing numbers after it
    # when move window right, remove from left side of q if no longer in window, pop and add on right side to maintain decreasing property
    '''                            Max      Queue (these are values but in algo below store indexes)
    [1]  3  -1 -3  5  3  6  7       3       [1]
    [1  3]  -1 -3  5  3  6  7       3       [3]
    [1  3  -1] -3  5  3  6  7       3       [3,-1]
     1 [3  -1  -3] 5  3  6  7       3       [3,-1,-3]
     1  3 [-1  -3  5] 3  6  7       5       [5]
     1  3  -1 [-3  5  3] 6  7       5       [5,3]
     1  3  -1  -3 [5  3  6] 7       6       [6]
     1  3  -1  -3  5 [3  6  7]      7       [7]
    '''
    result = []
    q = deque()
    for i in range(len(nums)):
        # add nums[i] to end of q maintaining monotonic decrease
        while q and nums[q[-1]] < nums[i]:
            q.pop()
        q.append(i)

        # remove from front of q if window doesn't include that number anymore
        lo = i-k+1  # i.e. if window ends at index 2 and has size 3 then 0 is the first index of window
        if q[0] < lo:
            q.popleft()
        if lo >= 0: # window is fully formed
            result.append(nums[q[0]])
    return result

# https://leetcode.com/problems/find-median-from-data-stream
# using builtin: O(1) findMedian and O(log(n)) addNum, O(n) space
class MedianFinderBuiltin:
    def __init__(self):
        self.sList = SortedList()

    def addNum(self, num: int) -> None:
        self.sList.add(num)

    def findMedian(self) -> float:
        # [1] -> 1, [1,2] -> 1.5, [1,2,3] -> 2
        if len(self.sList) % 2 == 0:
            i = len(self.sList) // 2
            return (self.sList[i] + self.sList[i-1]) / 2
        return self.sList[len(self.sList) // 2]

# O(1) findMedian and O(log(n)) addNum, O(n) space
class MedianFinder:
    def __init__(self):
        self.minHeap = []  # Holds the larger half
        self.maxHeap = []  # Holds the smaller half (negated values for max heap behavior)

    def addNum(self, num: int) -> None:
        if not self.maxHeap or num <= -self.maxHeap[0]:
            heapq.heappush(self.maxHeap, -num)
        else:
            heapq.heappush(self.minHeap, num)
        # Balance heaps, keep heaps same size or maxheap 1 larger than minheap
        if len(self.maxHeap) > len(self.minHeap) + 1:
            heapq.heappush(self.minHeap, -heapq.heappop(self.maxHeap))
        elif len(self.minHeap) > len(self.maxHeap):
            heapq.heappush(self.maxHeap, -heapq.heappop(self.minHeap))

    def findMedian(self) -> float:
        if len(self.maxHeap) > len(self.minHeap):
            return -self.maxHeap[0]
        return (-self.maxHeap[0] + self.minHeap[0]) / 2 if self.minHeap else -1

class MedianFinderUgly:
    def __init__(self):
        self.minHeap = []  # min of the upper half
        self.maxHeap = []  # max of the lowest half

    def addNum(self, num: int) -> None:
        if not self.minHeap:
            heapq.heappush(self.minHeap, num)
        elif not self.maxHeap:
            # just hit a size of 2
            if self.minHeap[0] > num: # num belongs in the max heap of the lower half
                heapq.heappush(self.maxHeap, -num)
            else:
                heapq.heappush(self.maxHeap, -self.minHeap.pop())
                heapq.heappush(self.minHeap, num)
        else:
            if len(self.maxHeap) > len(self.minHeap):
                if num >= self.minHeap[0]: # put num in the smaller sized min heap of largest nums
                    heapq.heappush(self.minHeap, num)
                else: # num goes in the larger sized max heap of smallest numbers (remove largest of these)
                    heapq.heappush(self.maxHeap, -num)
                    heapq.heappush(self.minHeap, -heapq.heappop(self.maxHeap))
            else:
                if num <= -self.maxHeap[0]: # put num in the smaller sized min heap of largest nums
                    heapq.heappush(self.maxHeap, -num)
                else: # num goes in the larger sized max heap of smallest numbers (remove largest of these)
                    heapq.heappush(self.minHeap, num)
                    heapq.heappush(self.maxHeap, -heapq.heappop(self.minHeap))

    def findMedian(self) -> float:
        if not self.minHeap:
            return -1
        if not self.maxHeap:
            return self.minHeap[0]
        if len(self.maxHeap) > len(self.minHeap):
            return -self.maxHeap[0]
        if len(self.maxHeap) < len(self.minHeap):
            return self.minHeap[0]
        return (-self.maxHeap[0] + self.minHeap[0]) / 2

# https://leetcode.com/problems/find-beautiful-indices-in-the-given-array-i
# O(n*m+n*n) time and O(n) space
def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
    def findAll(s, b):
        indices = []
        index = s.find(b)
        while index != -1:
            indices.append(index)
            index = s.find(b, index + 1)
        return indices

    result = []
    aIdxs = findAll(s, a)
    bIdxs = findAll(s, b)
    for aIdx in aIdxs:
        for bIdx in bIdxs:
            if abs(aIdx - bIdx) <= k:
                result.append(aIdx)
                break
    return result

# https://leetcode.com/problems/construct-quad-tree
# Definition for a QuadTree node.
class QuadNode:
    def __init__(self, val, isLeaf, topLeft=None, topRight=None, bottomLeft=None, bottomRight=None):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight
# O(N^2 *log N) time, O(log N) space for recursion stack
def construct(self, grid: List[List[int]]) -> 'QuadNode':
    def same(grid, x, y, size):
        expected = grid[y][x]
        for i in range(y, y + size):
            for j in range(x, x + size):
                if grid[i][j] != expected:
                    return False
        return True

    def helper(grid, x, y, size):
        if same(grid, x, y, size):
            return QuadNode(grid[y][x], True, None, None, None, None)
        else:
            half = size // 2
            ul = helper(grid, x, y, half)
            ur = helper(grid, x + half, y, half)
            bl = helper(grid, x, y + half, half)
            br = helper(grid, x + half, y + half, half)
            return QuadNode(grid[y][x], False, ul, ur, bl, br)

    return helper(grid, 0, 0, len(grid))
# O(N^2) time - avoids the redundant "same" checks by recursing first and combining same nodes going up
def constructBetter(self, grid):
    def solve(self, grid, x1, y1, length):
        if length == 1:
            return QuadNode(grid[x1][y1] == 1, True)

        half = length // 2
        topLeft = solve(grid, x1, y1, half)
        topRight = solve(grid, x1, y1 + half, half)
        bottomLeft = solve(grid, x1 + half, y1, half)
        bottomRight = solve(grid, x1 + half, y1 + half, half)

        # If the four returned nodes are leaf and have the same values, return a leaf node with that value.
        if (topLeft.isLeaf and topRight.isLeaf and bottomLeft.isLeaf and bottomRight.isLeaf and
                topLeft.val == topRight.val == bottomLeft.val == bottomRight.val):
            return QuadNode(topLeft.val, True)

        # If the four nodes aren't identical, return a non-leaf node with corresponding child pointers.
        return QuadNode(False, False, topLeft, topRight, bottomLeft, bottomRight)

    return solve(grid, 0, 0, len(grid))

# https://leetcode.com/problems/find-players-with-zero-or-one-losses
# O(n+2*nlogn) = O(nlogn) time and O(n) space,  could also consider counting sort for O(n+m) time
def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
    zeroLost = set()
    oneLost = set()
    moreLost = set()
    for m in matches:
        winner, loser = m[0], m[1]
        if winner not in moreLost and winner not in oneLost:
            zeroLost.add(winner)
        if loser not in moreLost:
            if loser in oneLost:
                oneLost.remove(loser)
                moreLost.add(loser)
            elif loser in zeroLost:
                zeroLost.remove(loser)
                oneLost.add(loser)
            else:
                oneLost.add(loser)
    return [sorted(list(zeroLost)), sorted(list(oneLost))]
# Alternative (same time and space complexity):
def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
    losses_count = {}
    for winner, loser in matches:
        losses_count[winner] = losses_count.get(winner, 0)
        losses_count[loser] = losses_count.get(loser, 0) + 1
    zero_lose, one_lose = [], []
    for player, count in losses_count.items():
        if count == 0:
            zero_lose.append(player)
        if count == 1:
            one_lose.append(player)
    return [sorted(zero_lose), sorted(one_lose)]

# https://leetcode.com/problems/number-of-divisible-triplet-sums
# O(n^2) avg O(n^3) worst case time, O(n^3) worst case space
def divisibleTripletCount(self, nums: List[int], d: int) -> int:
    n = len(nums)
    modVal = defaultdict(set)  # value mod d -> set(indexes with it)
    for i, num in enumerate(nums):
        modVal[num % d].add(i)
    r = set()

    for i in range(n - 1):
        for j in range(i + 1, n):
            m = (nums[i] + nums[j]) % d
            vals = modVal[(d - m) % d]
            for c in vals:
                if c == i or c == j:
                    continue
                r.add(tuple(sorted([i, j, c])))
    return len(r)
# O(n^3) time and O(1) space
def divisibleTripletCountBrute(self, nums: List[int], d: int) -> int:
    n = len(nums)
    r = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                if (nums[i] + nums[j] + nums[k]) % d == 0:
                    r += 1
    return r
# O(n^2) time and space
def divisibleTripletCountBetter(self, nums: List[int], d: int) -> int:
    mods = defaultdict(list)
    for i in range(len(nums) - 1):
        for j in range(i + 1, len(nums)):
            mod = (nums[i] + nums[j]) % d
            mods[mod].append((i, j))
    r = 0
    for k in range(2, len(nums)):
        target = (d - (nums[k] % d)) % d  # find pairs with a mod-sum that sums to d with current number mod; % by d to get 0 instead of d if target is d
        for pair in mods[target]:
            if pair[1] < k:  # only consider pairs before the current number
                r += 1
    return r
# O(n^2) time and O(1) space
def divisibleTripletCountBest(self, nums: List[int], d: int) -> int:
    n = len(nums)
    counter = Counter()
    res = 0
    # Iterate over all i,j pairs
    for j in range(n-2, 0, -1):
        counter[nums[j+1] % d] += 1 # counter stores count of k (highest val in triplet) % d
        # i is the lowest val in triplet, check all possible values for the current j
        for i in range(j):
            target = (d - (nums[i] + nums[j]) % d) % d
            res += counter[target]
    return res

# https://leetcode.com/problems/minimum-time-difference
# O(nlogn) time and O(n) space
def findMinDifferenceBetter(self, timePoints: List[str]) -> int:
    minutes = []
    for t in timePoints:
        parts = t.split(':')
        minutes.append(int(parts[0]) * 60 + int(parts[1]))
    minutes.sort()
    r = float('inf')
    for i in range(0, len(timePoints)-1):
        if i == 0:
            r = min(r, 60 * 24 - minutes[-1] + minutes[i])
        r = min(r, minutes[i+1] - minutes[i])
    return r
# O(n) time and O(1) space
def findMinDifferenceBucket(self, timePoints: List[str]) -> int:
    # create buckets array for the times converted to minutes
    minutes = [False] * (24 * 60)
    for time in timePoints:
        h, m = map(int, time.split(":"))
        min_time = h * 60 + m
        if minutes[min_time]:
            return 0
        minutes[min_time] = True
    prevIndex = float("inf")
    firstIndex = float("inf")
    lastIndex = float("inf")
    ans = float("inf")
    # find differences between adjacent elements in sorted array
    for i in range(24 * 60):
        if minutes[i]:
            if prevIndex != float("inf"):
                ans = min(ans, i - prevIndex)
            prevIndex = i
            if firstIndex == float("inf"):
                firstIndex = i
            lastIndex = i
    return min(ans, 24 * 60 - lastIndex + firstIndex)
# O(n^2) time and O(1) space
def findMinDifferenceBrute(self, timePoints: List[str]) -> int:
    def tAbs(t1, t2):
        if t2 > t1:
            t1, t2 = t2, t1
        diff = t1 - t2
        otherDiff = 60 * 24 - t1 + t2  # distance from high to upper limit and from low to zero
        return min(diff, otherDiff)

    r = float('inf')
    l = len(timePoints)
    for i in range(l - 1):
        for j in range(i + 1, l):
            p1p = timePoints[i].split(':')
            p2p = timePoints[j].split(':')
            p1min = int(p1p[0]) * 60 + int(p1p[1])
            p2min = int(p2p[0]) * 60 + int(p2p[1])
            r = min(r, tAbs(p1min, p2min))
    return r

# https://leetcode.com/problems/next-greater-element-iii
# O(n) time and O(1) space, same as "next permutation"
def nextGreaterElement(self, n: int) -> int:
    # 1234
    # 1243
    # 1324
    # 1342
    # 1423
    # 1432
    # 2134
    # 2143
    # 2314
    # 2341
    # 2413
    # 2431
    # 3124
    l = list(str(n))
    for i in range(len(l) - 2, -1, -1):
        c = l[i]
        p = l[i + 1]
        # if c < p:  # everything before c is sorted in decreasing order
    # swap c with number to its right that is just barely bigger
    # sort the numbers on the right side now to get smallest val possible
    # since values on the right were decreasing by reversing them we make them increasing

    return -1

# https://leetcode.com/problems/flatten-nested-list-iterator
class NestedInteger:
   def isInteger(self) -> bool:
       pass
       """
       @return True if this NestedInteger holds a single integer, rather than a nested list.
       """

   def getInteger(self) -> int:
       pass
       """
       @return the single integer that this NestedInteger holds, if it holds a single integer
       Return None if this NestedInteger holds a nested list
       """

   def getList(self) -> [NestedInteger]:
       pass
       """
       @return the nested list that this NestedInteger holds, if it holds a nested list
       Return None if this NestedInteger holds a single integer
       """
# extract everything on init O(n) and index through the big list O(1) using O(n) space
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.contents = []
        self.i = -1

        def extract(ni):
            if ni.isInteger():
                self.contents.append(ni.getInteger())
            else:
                for nestedi in ni.getList():
                    extract(nestedi)

        for nestedi in nestedList:
            extract(nestedi)

    def next(self) -> int:
        self.i += 1
        return self.contents[self.i]

    def hasNext(self) -> bool:
        return self.i < len(self.contents) - 1
# O(n+l): (n=#numbers and l=#lists) worst case everything is top level and gets added to stack in constructor
class NestedIteratorBetter:
    def __init__(self, nestedList: [NestedInteger]):
        nestedList.reverse()
        self.stack = nestedList  # top of stack is always an int, not another nested list (could be empty list)
        self._addInt()

    def _addInt(self):  # digs for a real int value using iterative dfs terminating when top of stack holds the next int
        while self.stack and not self.stack[-1].isInteger():
            nis = self.stack.pop().getList()
            for ni in reversed(nis):  # must reverse this to get the elements on the left at the top of the stack for next pop
                self.stack.append(ni)

    def next(self) -> int:
        toReturn = self.stack.pop()
        self._addInt()
        return toReturn

    def hasNext(self) -> bool:
        return bool(self.stack)
# Reversing the lists to put them onto the stack can be an expensive operation, and it turns out it isn't necessary.
# Instead of pushing every item of a sub-list onto the stack, we can instead associate an index pointer with each sub-list, that keeps track of how far along that sub-list we are. Adding a new sub-list to the stack now becomes an O(1) operation instead of a O(lengthofsublist) one.
# The total time complexity across all method calls for using up the entire iterator remains the same, but work is only done when it's necessary, thus improving performance when we only use part of the iterator. This is a desirable property for an iterator.
class NestedIteratorEvenBetter:
    def __init__(self, nestedList: [NestedInteger]):
        self.stack = [[nestedList, 0]] # list and a current index in the list

    def make_stack_top_an_integer(self):
        while self.stack:
            current_list = self.stack[-1][0]
            current_index = self.stack[-1][1]
            # top list is used up, pop it and its index
            if len(current_list) == current_index:
                self.stack.pop()
                continue
            # if it's already an integer, don't do anything
            if current_list[current_index].isInteger():
                break
            # Otherwise, it must be a list. We need to increment the index
            # on the previous list, and add the new list to proceed with 1 step of DFS.
            new_list = current_list[current_index].getList()
            self.stack[-1][1] += 1  # Increment old index
            self.stack.append([new_list, 0])  # add new list to use next iteration

    def next(self) -> int:
        self.make_stack_top_an_integer()
        current_list = self.stack[-1][0]
        current_index = self.stack[-1][1]
        self.stack[-1][1] += 1  # increment index in current list past the known int
        return current_list[current_index].getInteger()

    def hasNext(self) -> bool:
        self.make_stack_top_an_integer()
        return len(self.stack) > 0

class NestedIteratorGenerator:
    def __init__(self, nestedList: [NestedInteger]):
        # Get a generator object from the generator function, passing in
        # nestedList as the parameter.
        self._generator = self._int_generator(nestedList)
        # All values are placed here before being returned.
        self._peeked = None

    # This is the generator function. It can be used to create generator
    # objects.
    def _int_generator(self, nested_list) -> "Generator[int]":
        # This code is the same as Approach 1. It's a recursive DFS.
        for nested in nested_list:
            if nested.isInteger():
                yield nested.getInteger()
            else:
                # We always use "yield from" on recursive generator calls.
                yield from self._int_generator(nested.getList())
        # Will automatically raise a StopIteration.

    def next(self) -> int:
        # Check there are integers left, and if so, then this will
        # also put one into self._peeked.
        if not self.hasNext(): return None
        # Return the value of self._peeked, also clearing it.
        next_integer, self._peeked = self._peeked, None
        return next_integer

    def hasNext(self) -> bool:
        if self._peeked is not None: return True
        try:  # Get another integer out of the generator.
            self._peeked = next(self._generator)
            return True
        except:  # The generator is finished so raised StopIteration.
            return False

# https://leetcode.com/problems/number-of-flowers-in-full-bloom
def fullBloomFlowers(self, flowers: List[List[int]], people: List[int]) -> List[int]:
    # goal: find the number of overlaps at each person-time
    ans = [0] * len(people)
    speople = sorted(people)
    flowers.sort(key=lambda x: x[0])
    order = {}  # person time -> number of flowers

    pi = 0  # speople index
    heap = []  # hold currently blooming flowers ordered by end time low to high
    for f in flowers:  # process start times
        if pi == len(speople):
            break
        time = f[0]  # update currently blooming flowers for this time

        while heap and heap[0] < time:  # some flowers stopped blooming before current flower started
            endTime = heapq.heappop(heap)
            while pi < len(speople) and endTime >= speople[pi]:  # person visited before previous flower ended
                # print(f'person {speople[pi]} visited before endtime {endTime}')
                order[speople[pi]] = len(heap) + 1
                pi += 1

        while pi < len(speople) and time > speople[pi]:  # person visited before current flower started
            # print(f'person {speople[pi]} visited before flower {f} started')
            order[speople[pi]] = len(heap)
            pi += 1

        heapq.heappush(heap, f[1])  # start current flower

        while pi < len(speople) and time == speople[pi]:  # person visited when current flower started
            # print(f'person {speople[pi]} visited when flower {f} started')
            order[speople[pi]] = len(heap)
            pi += 1

    while pi < len(speople) and heap:  # process remaining end times
        endTime = heapq.heappop(heap)
        # print(f'person {speople[pi]} visited before final endtime {endTime}')
        while pi < len(speople) and endTime >= speople[pi]:  # person visited before previous flower ended
            order[speople[pi]] = len(heap) + 1
            pi += 1

    # put ans in the same order as original non-sorted people!
    for i, p in enumerate(people):
        ans[i] = order[p] if p in order else 0
    return ans
def fullBloomFlowersPretty(self, flowers: List[List[int]], people: List[int]) -> List[int]:
    flowers.sort()
    sorted_people = sorted(people)
    dic = {}
    heap = []
    i = 0
    for person in sorted_people:
        while i < len(flowers) and flowers[i][0] <= person:
            heapq.heappush(heap, flowers[i][1])
            i += 1
        while heap and heap[0] < person:
            heapq.heappop(heap)

        dic[person] = len(heap)
    return [dic[x] for x in people]
# difference arry + binary search: Time complexity: O((n+m)⋅logn), Space complexity: O(n)
def fullBloomFlowers(self, flowers: List[List[int]], people: List[int]) -> List[int]:
    difference = SortedDict({0: 0})
    for start, end in flowers:
        difference[start] = difference.get(start, 0) + 1
        difference[end + 1] = difference.get(end + 1, 0) - 1
    positions = []
    prefix = []
    curr = 0
    for key, val in difference.items():
        positions.append(key)
        curr += val
        prefix.append(curr)
    ans = []
    for person in people:
        i = bisect_right(positions, person) - 1
        ans.append(prefix[i])
    return ans
# simpler binary search: Time complexity: O((n+m)⋅logn), Space complexity: O(n)
def fullBloomFlowers(self, flowers: List[List[int]], people: List[int]) -> List[int]:
    starts = []
    ends = []
    for start, end in flowers:
        starts.append(start)
        ends.append(end + 1)
    starts.sort()
    ends.sort()
    ans = []
    for person in people:
        i = bisect_right(starts, person)
        j = bisect_right(ends, person)
        ans.append(i - j)
    return ans

# https://leetcode.com/problems/employee-importance
def getImportance(self, employees: List['Employee'], id: int) -> int:
    idToEmp = {}

    def subSum(e):
        if e:
            s = e.importance
            for sub in e.subordinates:
                s += subSum(idToEmp[sub])
            return s
        else:
            return 0

    for e in employees:
        idToEmp[e.id] = e

    return subSum(idToEmp[id])
    return -1

# https://leetcode.com/problems/text-justification
def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
    r = []
    i = 0
    while i < len(words):
        q = deque()
        l = 0
        while i < len(words) and l < maxWidth:
            w = words[i]
            if q:
                l += 1  # preceeding space
            if l + len(w) <= maxWidth:
                q.append(w)
                l += len(w)
            else:
                break
            i += 1

        # put contents of q together
        gaps = len(q) - 1
        print('gaps', gaps)
        print(q)
        if gaps == 0:
            word = q.popleft()
            white = ' ' * (maxWidth - len(word))
            r.append(word + white)
        else:
            charSum = sum([len(w) for w in q])
            gapLow = (maxWidth - charSum) // gaps
            s = ''
            while len(q) > 1:
                s = q.pop() + s
                s = ' ' * gapLow + s
            white = maxWidth - len(s) - len(q[0])
            s = q.pop() + ' ' * white + s
            r.append(s)
    return r

# https://leetcode.com/problems/sqrtx
def mySqrt(x: int) -> int:
    hi = x
    lo = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid * mid <= x:
            lo = mid + 1 # mid is ans in == case and want to end with hi as the return val, given hi < lo on loop exit we set lo to be mid + 1 so high = mid in this case
        else:
            hi = mid - 1
    print('lo', lo, 'hi', hi)
    return hi # on termination of while loop, hi < lo and we want the number <= the true sqrt
# mySqrt(x) = 2*mySqrt(x/4)
# x << y  =  x*2^y      x >> y  =  x/2^y
def mySqrtBits(self, x: int) -> int:
    if x < 2:
        return x
    left = self.mySqrt(x >> 2) << 1
    right = left + 1
    return left if right * right > x else right

# https://leetcode.com/problems/integer-to-roman
def int_to_roman(num):
    roman_numerals = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]

    result = ""
    for value, symbol in roman_numerals:
        while num >= value:
            result += symbol
            num -= value

    return result

# https://leetcode.com/problems/k-closest-points-to-origin
# O(N) time and space
def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
    # Precompute the Euclidean distance for each point
    distances = [self.euclidean_distance(point) for point in points]
    # Create a reference list of point indices
    remaining = [i for i in range(len(points))]
    # Define the initial binary search range
    low, high = 0, max(distances)

    # Perform a binary search of the distances
    # to find the k closest points
    closest = []
    while k:
        mid = (low + high) / 2
        closer, farther = self.split_distances(remaining, distances, mid)
        if len(closer) > k:
            # If more than k points are in the closer distances
            # then discard the farther points and continue
            remaining = closer
            high = mid
        else:
            # Add the closer points to the answer array and keep
            # searching the farther distances for the remaining points
            k -= len(closer)
            closest.extend(closer)
            remaining = farther
            low = mid

    # Return the k closest points using the reference indices
    return [points[i] for i in closest]

def split_distances(self, remaining: List[int], distances: List[float],
                    mid: int) -> List[List[int]]:
    """Split the distances around the midpoint
    and return them in separate lists."""
    closer, farther = [], []
    for index in remaining:
        if distances[index] <= mid:
            closer.append(index)
        else:
            farther.append(index)
    return [closer, farther]

def euclidean_distance(self, point: List[int]) -> float:
    """Calculate and return the squared Euclidean distance."""
    return point[0] ** 2 + point[1] ** 2
# O(N*logk) time and O(k) space
def kClosestHeap(self, points: List[List[int]], k: int) -> List[List[int]]:
    # Since heap is sorted in increasing order,
    # negate the distance to simulate max heap
    # and fill the heap with the first k elements of points
    heap = [(-self.squared_distance(points[i]), i) for i in range(k)]
    heapq.heapify(heap)
    for i in range(k, len(points)):
        dist = -self.squared_distance(points[i])
        if dist > heap[0][0]:
            # If this point is closer than the kth farthest,
            # discard the farthest point and add this one
            heapq.heappushpop(heap, (dist, i))

    # Return all points stored in the max heap
    return [points[i] for (_, i) in heap]

def squared_distance(self, point: List[int]) -> int:
    """Calculate and return the squared Euclidean distance."""
    return point[0] ** 2 + point[1] ** 2

# https://leetcode.com/problems/valid-number
def isNumber(self, s: str) -> bool:
    seen_digit = seen_exponent = seen_dot = False
    for i, c in enumerate(s):
        if c.isdigit():
            seen_digit = True
        elif c in ["+", "-"]:
            if i > 0 and s[i - 1] != "e" and s[i - 1] != "E":
                return False
        elif c in ["e", "E"]:
            if seen_exponent or not seen_digit:
                return False
            seen_exponent = True
            seen_digit = False
        elif c == ".":
            if seen_dot or seen_exponent:
                return False
            seen_dot = True
        else:
            return False
    return seen_digit
def isNumberDFA(self, s):
    # This is the Deterministic Finite Automaton (DFA) we have designed above
    dfa = [
        {"digit": 1, "sign": 2, "dot": 3},
        {"digit": 1, "dot": 4, "exponent": 5},
        {"digit": 1, "dot": 3},
        {"digit": 4},
        {"digit": 4, "exponent": 5},
        {"sign": 6, "digit": 7},
        {"digit": 7},
        {"digit": 7},
    ]
    current_state = 0
    for c in s:
        if c.isdigit():
            group = "digit"
        elif c in ["+", "-"]:
            group = "sign"
        elif c in ["e", "E"]:
            group = "exponent"
        elif c == ".":
            group = "dot"
        else:
            return False
        if group not in dfa[current_state]:
            return False

        current_state = dfa[current_state][group]
    return current_state in [1, 4, 7]

class Node:
    def __init__(self, freq):
        self.freq = freq
        self.prev = None
        self.next = None
        self.keys = set()


class AllOne:
    def __init__(self):
        self.head = Node(0)  # Dummy head
        self.tail = Node(0)  # Dummy tail
        self.head.next = self.tail  # Link dummy head to dummy tail
        self.tail.prev = self.head  # Link dummy tail to dummy head
        self.map = {}  # Mapping from key to its node

    def inc(self, key: str) -> None:
        if key in self.map:
            node = self.map[key]
            freq = node.freq
            node.keys.remove(key)  # Remove key from current node

            nextNode = node.next
            if nextNode == self.tail or nextNode.freq != freq + 1:
                # Create a new node if next node does not exist or freq is not freq + 1
                newNode = Node(freq + 1)
                newNode.keys.add(key)
                newNode.prev = node
                newNode.next = nextNode
                node.next = newNode
                nextNode.prev = newNode
                self.map[key] = newNode
            else:
                # Increment the existing next node
                nextNode.keys.add(key)
                self.map[key] = nextNode

            # Remove the current node if it has no keys left
            if not node.keys:
                self.removeNode(node)
        else:  # Key does not exist
            firstNode = self.head.next
            if firstNode == self.tail or firstNode.freq > 1:
                # Create a new node
                newNode = Node(1)
                newNode.keys.add(key)
                newNode.prev = self.head
                newNode.next = firstNode
                self.head.next = newNode
                firstNode.prev = newNode
                self.map[key] = newNode
            else:
                firstNode.keys.add(key)
                self.map[key] = firstNode

    def dec(self, key: str) -> None:
        if key not in self.map:
            return  # Key does not exist

        node = self.map[key]
        node.keys.remove(key)
        freq = node.freq

        if freq == 1:
            # Remove the key from the map if freq is 1
            del self.map[key]
        else:
            prevNode = node.prev
            if prevNode == self.head or prevNode.freq != freq - 1:
                # Create a new node if the previous node does not exist or freq is not freq - 1
                newNode = Node(freq - 1)
                newNode.keys.add(key)
                newNode.prev = prevNode
                newNode.next = node
                prevNode.next = newNode
                node.prev = newNode
                self.map[key] = newNode
            else:
                # Decrement the existing previous node
                prevNode.keys.add(key)
                self.map[key] = prevNode

        # Remove the node if it has no keys left
        if not node.keys:
            self.removeNode(node)

    def getMaxKey(self) -> str:
        if self.tail.prev == self.head:
            return ""  # No keys exist
        return next(
            iter(self.tail.prev.keys)
        )  # Return one of the keys from the tail's previous node

    def getMinKey(self) -> str:
        if self.head.next == self.tail:
            return ""  # No keys exist
        return next(
            iter(self.head.next.keys)
        )  # Return one of the keys from the head's next node

    def removeNode(self, node):
        prevNode = node.prev
        nextNode = node.next

        prevNode.next = nextNode  # Link previous node to next node
        nextNode.prev = prevNode  # Link next node to previous node

# https://leetcode.com/problems/basic-calculator-ii
'''
class Solution {
    public int calculate(String s) {

        if (s == null || s.isEmpty()) return 0;
        int len = s.length();
        Stack<Integer> stack = new Stack<Integer>();
        int currentNumber = 0;
        char operation = '+';
        for (int i = 0; i < len; i++) {
            char currentChar = s.charAt(i);
            if (Character.isDigit(currentChar)) {
                currentNumber = (currentNumber * 10) + (currentChar - '0');
            }
            if (!Character.isDigit(currentChar) && !Character.isWhitespace(currentChar) || i == len - 1) {
                if (operation == '-') {
                    stack.push(-currentNumber);
                }
                else if (operation == '+') {
                    stack.push(currentNumber);
                }
                else if (operation == '*') {
                    stack.push(stack.pop() * currentNumber);
                }
                else if (operation == '/') {
                    stack.push(stack.pop() / currentNumber);
                }
                operation = currentChar;
                currentNumber = 0;
            }
        }
        int result = 0;
        while (!stack.isEmpty()) {
            result += stack.pop();
        }
        return result;
    }
}
# O(1) space:
class Solution {
    public int calculate(String s) {
        if (s == null || s.isEmpty()) return 0;
        int length = s.length();
        int currentNumber = 0, lastNumber = 0, result = 0;
        char operation = '+';
        for (int i = 0; i < length; i++) {
            char currentChar = s.charAt(i);
            if (Character.isDigit(currentChar)) {
                currentNumber = (currentNumber * 10) + (currentChar - '0');
            }
            if (!Character.isDigit(currentChar) && !Character.isWhitespace(currentChar) || i == length - 1) {
                if (operation == '+' || operation == '-') {
                    result += lastNumber;
                    lastNumber = (operation == '+') ? currentNumber : -currentNumber;
                } else if (operation == '*') {
                    lastNumber = lastNumber * currentNumber;
                } else if (operation == '/') {
                    lastNumber = lastNumber / currentNumber;
                }
                operation = currentChar;
                currentNumber = 0;
            }
        }
        result += lastNumber;
        return result;
    }
}
'''
def calculateUgly(self, s: str) -> int:
    s = "".join(s.split())  # remove all whitespace
    opStack = []
    numStack = []
    num = ""
    for c in s:
        if c.isdigit():
            if num:
                num += c
            else:
                num = c
        else:  # operation
            if opStack and opStack[-1] == '*':  # finished the 2nd num of multiplication
                numStack.append(numStack.pop() * int(num))
                opStack.pop()
            elif opStack and opStack[-1] == '/':  # finished the 2nd num of division
                numStack.append(numStack.pop() // int(num))
                opStack.pop()
            else:
                numStack.append(int(num))
            opStack.append(c)
            num = ""
    if opStack and opStack[-1] == '*':  # finished the 2nd num of multiplication
        numStack.append(numStack.pop() * int(num))
        opStack.pop()
    elif opStack and opStack[-1] == '/':  # finished the 2nd num of division
        numStack.append(numStack.pop() // int(num))
        opStack.pop()
    else:
        numStack.append(int(num))
    # complete addition and subtraction left to right
    if numStack:
        prev = numStack[0]
        opI = 0
        for i in range(1, len(numStack)):
            if opStack[opI] == '-':
                prev -= numStack[i]
            else:
                prev += numStack[i]
            opI += 1
        return prev
    else:
        return numStack[-1]

# https://leetcode.com/problems/time-based-key-value-store
from sortedcontainers import SortedDict
class TimeMap:
    def __init__(self):
        self.key_time_map = {}

    def set(self, key: str, value: str, timestamp: int) -> None:
        # If the 'key' does not exist in dictionary.
        if not key in self.key_time_map:
            self.key_time_map[key] = SortedDict()

        # Store '(timestamp, value)' pair in 'key' bucket.
        self.key_time_map[key][timestamp] = value

    def get(self, key: str, timestamp: int) -> str:
        # If the 'key' does not exist in dictionary we will return empty string.
        if not key in self.key_time_map:
            return ""

        it = self.key_time_map[key].bisect_right(timestamp)
        # If iterator points to first element it means, no time <= timestamp exists.
        if it == 0:
            return ""

        # Return value stored at previous position of current iterator.
        return self.key_time_map[key].peekitem(it - 1)[1]
class TimeMap2:
    def __init__(self):
        self.key_time_map = {}

    def set(self, key: str, value: str, timestamp: int) -> None:
        # If the 'key' does not exist in dictionary.
        if not key in self.key_time_map:
            self.key_time_map[key] = []

        # Store '(timestamp, value)' pair in 'key' bucket.
        self.key_time_map[key].append([timestamp, value])

    def get(self, key: str, timestamp: int) -> str:
        # If the 'key' does not exist in dictionary we will return empty string.
        if not key in self.key_time_map:
            return ""

        if timestamp < self.key_time_map[key][0][0]:
            return ""

        left = 0
        right = len(self.key_time_map[key])

        while left < right:
            mid = (left + right) // 2
            if self.key_time_map[key][mid][0] <= timestamp:
                left = mid + 1
            else:
                right = mid

        # If iterator points to first element it means, no time <= timestamp exists.
        return "" if right == 0 else self.key_time_map[key][right - 1][1]

# https://leetcode.com/problems/maximum-profit-in-job-scheduling
# O(nlogn) time and O(n) space
'''
class Solution {
    // maximum number of jobs are 50000
    int[] memo = new int[50001];
    
    private int findNextJob(int[] startTime, int lastEndingTime) {
        int start = 0, end = startTime.length - 1, nextIndex = startTime.length;
        
        while (start <= end) {
            int mid = (start + end) / 2;
            if (startTime[mid] >= lastEndingTime) {
                nextIndex = mid;
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }
        return nextIndex;
    }
    
    private int findMaxProfit(List<List<Integer>> jobs, int[] startTime, int n, int position) {
        // 0 profit if we have already iterated over all the jobs
        if (position == n) {
            return 0;
        }
        
        // return result directly if it's calculated 
        if (memo[position] != -1) {
            return memo[position];
        }
        
        // nextIndex is the index of next non-conflicting job
        int nextIndex = findNextJob(startTime, jobs.get(position).get(1));
        
        // find the maximum profit of our two options: skipping or scheduling the current job
        int maxProfit = Math.max(findMaxProfit(jobs, startTime, n, position + 1), 
                        jobs.get(position).get(2) + findMaxProfit(jobs, startTime, n, nextIndex));
        
        // return maximum profit and also store it for future reference (memoization)
        return memo[position] = maxProfit;
    }
    
    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
        List<List<Integer>> jobs = new ArrayList<>();
        
        // marking all values to -1 so that we can differentiate 
        // if we have already calculated the answer or not
        Arrays.fill(memo, -1);
        
        // storing job's details into one list 
        // this will help in sorting the jobs while maintaining the other parameters
        int length = profit.length;
        for (int i = 0; i < length; i++) {
            ArrayList<Integer> currJob = new ArrayList<>();
            currJob.add(startTime[i]);
            currJob.add(endTime[i]);
            currJob.add(profit[i]);
            jobs.add(currJob);
        }
        jobs.sort(Comparator.comparingInt(a -> a.get(0)));
        
        // binary search will be used in startTime so store it as separate list
        for (int i = 0; i < length; i++) {
            startTime[i] = jobs.get(i).get(0);
        }
        
        return findMaxProfit(jobs, startTime, length, 0);
    }
}
class Solution {
    class The_Comparator implements Comparator<ArrayList<Integer>> {
        public int compare(ArrayList<Integer> list1, ArrayList<Integer> list2) {
            return list1.get(0) - list2.get(0);
        }
    }
    
    private int findMaxProfit(List<List<Integer>> jobs) {
        int n = jobs.size(), maxProfit = 0;
        // min heap having {endTime, profit}
        PriorityQueue<ArrayList<Integer>> pq = new PriorityQueue<>(new The_Comparator());
        
        for (int i = 0; i < n; i++) {
            int start = jobs.get(i).get(0), end = jobs.get(i).get(1), profit = jobs.get(i).get(2);
            
            // keep popping while the heap is not empty and
            // jobs are not conflicting
            // update the value of maxProfit
            while (pq.isEmpty() == false && start >= pq.peek().get(0)) {
                maxProfit = Math.max(maxProfit, pq.peek().get(1));
                pq.remove();
            }
            
            ArrayList<Integer> combinedJob = new ArrayList<>();
            combinedJob.add(end);
            combinedJob.add(profit + maxProfit);
            
            // push the job with combined profit
            // if no non-conflicting job is present maxProfit will be 0
            pq.add(combinedJob);
        }
        
        // update the value of maxProfit by comparing with
        // profit of jobs that exits in the heap
        while (pq.isEmpty() == false) {
            maxProfit = Math.max(maxProfit, pq.peek().get(1));
            pq.remove();
        }
        
        return maxProfit;
    }
    
    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
        List<List<Integer>> jobs = new ArrayList<>();
        
        // storing job's details into one list 
        // this will help in sorting the jobs while maintaining the other parameters
        int length = profit.length;
        for (int i = 0; i < length; i++) {
            ArrayList<Integer> currJob = new ArrayList<>();
            currJob.add(startTime[i]);
            currJob.add(endTime[i]);
            currJob.add(profit[i]);
            
            jobs.add(currJob);
        }
        
        jobs.sort(Comparator.comparingInt(a -> a.get(0)));
        return findMaxProfit(jobs);
    }
}
'''