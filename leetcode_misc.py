import heapq
from collections import deque, Counter
from typing import List, Optional




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


