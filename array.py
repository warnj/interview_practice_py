import bisect
import collections
import heapq
import random
from bisect import bisect_left
from collections import deque, Counter
import sys
import math
from typing import List


class ArrayPractice:
    # https://leetcode.com/problems/maximum-sum-circular-subarray
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        curSum, maxSum = 0, -sys.maxsize
        sums = 0
        for i in range(len(nums)):
            sums += nums[i]
            curSum += nums[i]
            maxSum = max(curSum, maxSum)
            if curSum < 0:
                curSum = 0
        curSum = 0
        max2 = 0
        for i in range(len(nums)):
            maxSum = max(maxSum, sums + max2)
            sums -= nums[i]
            curSum += nums[i]
            max2 = max(max2, curSum)
        return maxSum

    # returns the lowest int possible that is greater than lo from swapping any one pair of digits in tgt
    def __getSwap(self, lo, tgt) -> int:
        result = None
        num = list(map(int, list(str(tgt))))
        for i in range(len(num)-1):
            for j in range(i+1, len(num)):
                temp = num.copy()
                temp[i] = num[j]
                temp[j] = num[i]
                tInt = int("".join(str(x) for x in temp))
                if (not result and tInt > lo) or (result and result > tInt > lo):
                    result = tInt
        return result

    # given array of non-neg numbers, can choose any number from the array and swap any 2 digits
    # return if it's possible to apply swap at most once so elements are strictly increasing
    def swapOnceIncr(self, nums):
        swapped = False
        prevPrev = -sys.maxsize
        for i in range(1, len(nums)):
            prev = nums[i-1]
            cur = nums[i]
            if prev >= cur:
                if swapped:  # already done a swap, it's only allowed once
                    return False
                else:
                    s = self.__getSwap(prevPrev, prev)
                    if s:
                        swapped = True
                        # print('swapping {} for {}'.format(prev, s))
                        prev = s
                        nums[i-1] = s
                    else:
                        return False
            prevPrev = prev
        return True

    # given array of positive ints return sum of every possible string concatenation of a[i] and a[j]
    # example: [1,2,3] = 11+12+13+21+22+23+31+32+33=198
    def concatSum(self, nums: List[int]) -> int:
        lowSum = 0
        offsetSum = 0
        for i in range(len(nums)):
            lowSum += nums[i]
            size = len(str(nums[i]))
            offset = 10**size
            offsetSum += offset
        return lowSum * len(nums) + lowSum * offsetSum
    def concatSumSlowInts(self, nums: List[int]) -> int:
        result = 0
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                result += self.__concatInts(nums[i], nums[j])
                if i != j:
                    result += self.__concatInts(nums[j], nums[i])
        return result
    def concatSumSlow(self, nums: List[int]) -> int:
        result = 0
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                a = str(nums[i])
                b = str(nums[j])
                result += int(a+b) if i == j else int(a+b) + int(b+a)
        return result
    def concatSumSlowOG(self, nums: List[int]) -> int:
        result = 0
        for i in range(len(nums)):
            for j in range(len(nums)):
                print('i={} j={}'.format(i, j))
                result += int(str(nums[i]) + str(nums[j]))
        return result
    def __concatInts(self, x, y):
        if y != 0:
            a = math.floor(math.log10(y))
        else:
            a = -1
        return int(x*10**(1+a)+y)

    # https://leetcode.com/problems/longest-common-prefix
    # worst case n^2 time, can do better with sorting (compare most dissimilar strings 1st and last ones)
    def longestCommonPrefix(self, strs: List[str]) -> str:
        result = ''
        while True:
            if len(result) == len(strs[0]):
                return result
            else:
                char = strs[0][len(result)]
                for i in range(1, len(strs)):
                    s = strs[i]
                    if len(result) == len(s):
                        return result
                    elif char != s[len(result)]:
                        return result
                result += char

    # there is a better way than this brute force solution, consider sorting or building a trie with first list and comparing to 2nd list
    def longestCommonPrefixTwoLists(self, strs1, strs2):
        result = -sys.maxsize
        for s1 in strs1:
            for s2 in strs2:
                result = max(result, self.__getCommonPrefix(s1, s2))
        return result
    def __getCommonPrefix(self, s1, s2):
        l = 0
        for i in range(min(len(s1), len(s2))):
            if s1[i] != s2[i]:
                return l
            l += 1
        return l


def main() -> None:
    a = ArrayPractice()
    # print(a.maxSubarraySumCircular([1, -2, 3, -2]))
    # print(a.swapOnceIncr([1, 3, 9, 10]))


if __name__ == "__main__":
    main()


# https://leetcode.com/problems/kth-largest-element-in-an-array/
# O(nlogk) time and O(k) space
def findKthLargest(self, nums: List[int], k: int) -> int:
    heap = nums[:k]  # store the k largest in a min heap
    heapq.heapify(heap)
    for i in range(k, len(nums)):
        heapq.heappushpop(heap, nums[i])  # adds current elem and remove the smallest to keep the k largest in heap
    return heap[0]  # kth smallest is the top of the k largest
# pick random partition index, move smaller elements to left and larger elements to right, if partition is k return the value or search left/right of it
# O(n) avg and best time, O(n^2) worst, O(1) space
def findKthLargestQuickSelect(nums, k):
    left, right = 0, len(nums) - 1
    while True:
        pivot_index = random.randint(left, right)
        new_pivot_index = partition(nums, left, right, pivot_index)
        if new_pivot_index == len(nums) - k:
            return nums[new_pivot_index]
        elif new_pivot_index > len(nums) - k:
            right = new_pivot_index - 1
        else:
            left = new_pivot_index + 1
def partition(nums, left, right, pivot_index):
    pivot = nums[pivot_index]
    nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
    stored_index = left
    for i in range(left, right):
        if nums[i] < pivot:
            nums[i], nums[stored_index] = nums[stored_index], nums[i]
            stored_index += 1
    nums[right], nums[stored_index] = nums[stored_index], nums[right]
    return stored_index
def findKthLargestCountingSort(self, nums: List[int], k: int) -> int:
    min_value = min(nums)
    max_value = max(nums)
    count = [0] * (max_value - min_value + 1)
    for num in nums:
        count[num - min_value] += 1
    remain = k
    for num in range(len(count) - 1, -1, -1):
        remain -= count[num]
        if remain <= 0:
            return num + min_value
    return -1

# https://leetcode.com/problems/target-sum
# backtracing O(2^n) time (2 recursive choices for each of n elements) and O(n) space (depth of recursion stack)
def findTargetSumWays(self, nums: List[int], target: int) -> int:
    def findTargetSum(i, curSum):
        if i == len(nums):
            if curSum == target:
                return 1
            return 0
        else:
            return findTargetSum(i + 1, curSum + nums[i]) + findTargetSum(i + 1, curSum - nums[i])
    return findTargetSum(0, 0)
# O(n*totalsum) time and space
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        self.total_sum = sum(nums)
        memo = [
            [float("-inf")] * (2 * self.total_sum + 1) for _ in range(len(nums))
        ]
        return self.calculate_ways(nums, 0, 0, target, memo)

    def calculate_ways(
        self,
        nums: List[int],
        current_index: int,
        current_sum: int,
        target: int,
        memo: List[List[int]],
    ) -> int:
        if current_index == len(nums):
            # Check if the current sum matches the target
            return 1 if current_sum == target else 0
        else:
            # Check if the result is already computed
            if memo[current_index][current_sum + self.total_sum] != float(
                "-inf"
            ):
                return memo[current_index][current_sum + self.total_sum]

            # Calculate ways by adding the current number
            add = self.calculate_ways(
                nums,
                current_index + 1,
                current_sum + nums[current_index],
                target,
                memo,
            )

            # Calculate ways by subtracting the current number
            subtract = self.calculate_ways(
                nums,
                current_index + 1,
                current_sum - nums[current_index],
                target,
                memo,
            )

            # Store the result in memoization table
            memo[current_index][current_sum + self.total_sum] = add + subtract

            return memo[current_index][current_sum + self.total_sum]
# https://leetcode.com/problems/target-sum/editorial/?envType=daily-question&envId=2024-12-26
# O(n*totalsum) time and space
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total_sum = sum(nums)
        dp = [[0] * (2 * total_sum + 1) for _ in range(len(nums))]

        # Initialize the first row of the DP table
        dp[0][nums[0] + total_sum] = 1
        dp[0][-nums[0] + total_sum] += 1

        # Fill the DP table
        for index in range(1, len(nums)):
            for sum_val in range(-total_sum, total_sum + 1):
                if dp[index - 1][sum_val + total_sum] > 0:
                    dp[index][sum_val + nums[index] + total_sum] += dp[
                        index - 1
                    ][sum_val + total_sum]
                    dp[index][sum_val - nums[index] + total_sum] += dp[
                        index - 1
                    ][sum_val + total_sum]

        # Return the result if the target is within the valid range
        return (
            0
            if abs(target) > total_sum
            else dp[len(nums) - 1][target + total_sum]
        )

# https://leetcode.com/problems/best-sightseeing-pair
# O(n) time O(1) space
def maxScoreSightseeingPair(self, values):
    n = len(values)
    max_left_score = values[0]
    max_score = 0
    for i in range(1, n):
        current_right_score = values[i] - i
        max_score = max(max_score, max_left_score + current_right_score)
        current_left_score = values[i] + i
        max_left_score = max(max_left_score, current_left_score)
    return max_score
# O(n) time and space
def maxScoreSightseeingPair2(self, values: List[int]) -> int:
    # fill in the value at each index if it was used as i or j
    jVals = []
    for i, v in enumerate(values):
        jVals.append(v-i)
    iVals = []
    for i, v in enumerate(values):
        iVals.append(v+i)
    # transform to the max i value seen before a given index or max j value seen after
    for i in range(len(values)-2, -1, -1):
        jVals[i] = max(jVals[i], jVals[i+1])  # jMax[i] = max jval >= i
    for i in range(1, len(values)):
        iVals[i] = max(iVals[i], iVals[i-1])

    result = float('-inf')
    for i in range(0, len(values)-1):
        result = max(result, iVals[i] + jVals[i+1])  # i < j so look 1 ahead for max j
    return result
# O(n^2) time and O(1) space
def maxScoreSightseeingPairBrute(self, values: List[int]) -> int:
    result = float('-inf')
    for i in range(0, len(values)-1):
        for j in range(i+1, len(values)):
            result = max(result, values[i] + values[j] + i - j)
    return result

# https://leetcode.com/problems/find-peak-element
# O(log n) time and O(1) space
def findPeakElement(self, nums: List[int]) -> int:
    l = 0
    r = len(nums) - 1
    while l < r:
        mid = (l + r) // 2
        if nums[mid] > nums[mid + 1]:
            r = mid
        else:
            l = mid + 1
    return l
def findPeakElementOG(self, nums: List[int]) -> int:
    def isPeak(i, nums):
        n = len(nums)
        if i == n - 1:  # last element
            if nums[i - 1] < nums[i]:
                return True
            else:
                return False
        if i == 0:  # first element
            if nums[i] > nums[1]:
                return True
            else:
                return False
        return nums[i - 1] < nums[i] > nums[i + 1]
    if len(nums) == 1:
        return 0
    if len(nums) == 2:
        if nums[0] > nums[1]:
            return 0
        else:
            return 1
    hi = len(nums) - 1
    lo = 0
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if isPeak(mid, nums):
            return mid
        if nums[mid + 1] > nums[mid]:
            lo = mid + 1  # go uphill
        else:
            hi = mid
    return hi

# https://leetcode.com/problems/top-k-frequent-elements
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    counts = {}
    for n in nums:
        counts[n] = counts.get(n, 0) + 1

    heap = []
    for n, c in counts.items():
        heap.append((-c, n))
    heapq.heapify(heap)

    result = []
    for i in range(k):
        result.append(heapq.heappop(heap)[1])
    return result

# https://leetcode.com/problems/top-k-frequent-words
# O(N*logk) time and O(N+k) space (for counts + heap respectively)
class Pair:
    def __init__(self, word, freq):
        self.word = word
        self.freq = freq
    def __lt__(self, p): # use for max heap behavior with Python's min heap operations
        return self.freq < p.freq or (self.freq == p.freq and self.word > p.word)
def topKFrequentBest(self, words: List[str], k: int) -> List[str]:
    cnt = Counter(words)
    h = []
    for word, freq in cnt.items():
        if len(h) == k:
            heapq.heappushpop(h, Pair(word, freq))
        else:
            heapq.heappush(h, Pair(word, freq))
    return [p.word for p in sorted(h, reverse=True)]
# O(n + klogn) time and O(n) space
def topKFrequent(self, words: List[str], k: int) -> List[str]:
    counts = Counter(words) # map word -> count
    heap = [(-freq, word) for word, freq in counts.items()]
    heapq.heapify(heap) # O(n)
    return [heapq.heappop(heap)[1] for _ in range(k)] # remove top k from heap: O(k*logn)

# https://leetcode.com/problems/subarray-sum-equals-k
# if the cumulative sum up to two indices, say i and j is at a difference of k i.e. if sum[i]−sum[j]=k, the sum of elements lying between indices i and j is k
# O(n) time and space
def subarraySum(self, nums: List[int], k: int) -> int:
    count = 0
    total_sum = 0
    prefix_sum_map = {0: 1} # sum -> num of occurrences of sum
    for num in nums:
        total_sum += num
        if total_sum - k in prefix_sum_map:
            count += prefix_sum_map[total_sum - k]
        prefix_sum_map[total_sum] = prefix_sum_map.get(total_sum, 0) + 1  # increment sum count by 1
    return count
# O(n^2) time O(1) space
def subarraySumBrute(self, nums: List[int], k: int) -> int:
    r = 0
    for lo in range(len(nums)):
        s = nums[lo]
        if s == k:
            r += 1
        for hi in range(lo + 1, len(nums)):
            s += nums[hi]
            if s == k:
                r += 1
    return r

# https://leetcode.com/problems/count-vowel-strings-in-ranges
# O(w + sum(ranges in queries)) time and O(w) space
def vowelStringsBrute(self, words: List[str], queries: List[List[int]]) -> List[int]:
    def isVowel(char):
        return char == 'a' or char == 'e' or char == 'i' or char == 'o' or char == 'u'
    startEndVowel = {}
    for i, w in enumerate(words):
        startEndVowel[i] = isVowel(w[0]) and isVowel(w[-1])
    result = []
    for q in queries:
        r = 0
        for i in range(q[0], q[1] + 1):
            if startEndVowel[i]:
                r += 1
        result.append(r)
    return result
# O(w+q) time and O(w) space
def vowelStrings(self, words: List[str], queries: List[List[int]]) -> List[int]:
    def isVowel(char):
        return char == 'a' or char == 'e' or char == 'i' or char == 'o' or char == 'u'
    dp = [0] * len(words)  # dp[i] = number of words that start and end with vowel from [0,i] inclusive
    for i, w in enumerate(words):
        prev = 0 if i == 0 else dp[i - 1]
        cur = 1 if isVowel(w[0]) and isVowel(w[-1]) else 0
        dp[i] = prev + cur

    result = []
    for q in queries:
        if q[0] == 0:
            result.append(dp[q[1]])
        else:
            result.append(dp[q[1]] - dp[q[0] - 1]) # bottom index is inclusive so must subtract cumulative sum before it
    return result

# https://leetcode.com/problems/number-of-ways-to-split-array
def waysToSplitArray(self, nums: list[int]) -> int:
    # Keep track of sum of elements on left and right sides
    left_sum = right_sum = 0
    # Initially all elements are on right side
    right_sum = sum(nums)
    # Try each possible split position
    count = 0
    for i in range(len(nums) - 1):
        # Move current element from right to left side
        left_sum += nums[i]
        right_sum -= nums[i]
        # Check if this creates a valid split
        if left_sum >= right_sum:
            count += 1
    return count
def waysToSplitArrayExtraMemory(self, nums: List[int]) -> int:
    prefixSums = [0] * len(nums) # prefixSums[i] is the sum to i in nums inclusive
    suffixSums = [0] * len(nums) # suffixSums[i] is the sum after i in nums inclusive

    prefixSums[0] = nums[0]
    for i in range(1, len(nums)):
        prefixSums[i] = prefixSums[i-1] + nums[i]
    suffixSums[-1] = nums[-1]
    for i in range(len(nums)-2, -1, -1):
        suffixSums[i] = suffixSums[i+1] + nums[i]

    r = 0
    for i in range(len(nums)-1):
        if prefixSums[i] >= suffixSums[i+1]:
            r += 1
    return r

# https://leetcode.com/problems/two-sum/
def twoSum(self, nums: List[int], target: int) -> List[int]:
    seen = {}
    for i, n in enumerate(nums):
        want = target - n
        if want in seen:
            return seen[want], i
        seen[n] = i
    return None

# https://leetcode.com/problems/3sum/
# O(n^2) time and O(n)
def threeSumOG(self, nums: List[int]) -> List[List[int]]:
    result = set()  # used to avoid duplicates
    for i in range(0, len(nums) - 2):
        target = -nums[i]
        # run two sum looking for numbers summing to target
        vals = set()
        for j in range(i + 1, len(nums)):
            complement = target - nums[j]
            if complement in vals:
                result.add(tuple(sorted([nums[i], nums[j], complement])))
            vals.add(nums[j])
    finalResult = []
    for r in result:
        finalResult.append(list(r))
    return finalResult
# O(n^2) time and O(n) space
def threeSumBest(self, nums: List[int]) -> List[List[int]]:
    res = []
    nums.sort()
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i - 1]:  # already looked for triples starting with this value
            continue
        # run two sum on a sorted array
        lo = i + 1
        hi = len(nums) - 1
        while lo < hi:
            total = nums[i] + nums[lo] + nums[hi]
            if total > 0:
                hi -= 1
            elif total < 0:
                lo += 1
            else:
                res.append([nums[i], nums[lo], nums[hi]])
                lo += 1
                while nums[lo] == nums[lo - 1] and lo < hi:  # optimization to avoid duplicate middle numbers
                    lo += 1
    return res
def threeSum2(self, nums: List[int]) -> List[List[int]]:
    res = set()
    # split nums into three lists: negative, positive, and zero
    n, p, z = [], [], []
    for num in nums:
        if num > 0:
            p.append(num)
        elif num < 0:
            n.append(num)
        else:
            z.append(num)
    N, P = set(n), set(p)
    # If there is a zero in the list, add all cases where -num exists in N and num exists in P
    if z:
        for num in P:
            if -1*num in N:
                res.add((-1*num, 0, num))
    if len(z) >= 3:
        res.add((0,0,0))
    # For all pairs of negative numbers (-3, -1), check to see if their complement exists in the positives
    for i in range(len(n)):
        for j in range(i+1,len(n)):
            target = -1*(n[i]+n[j])
            if target in P:
                res.add(tuple(sorted([n[i],n[j],target])))
    # vice versa for pairs of positive numbers
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            target = -1*(p[i]+p[j])
            if target in N:
                res.add(tuple(sorted([p[i],p[j],target])))
    return res

# https://leetcode.com/problems/merge-sorted-array
def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    n1 = m - 1
    n2 = n - 1
    i = n + m - 1
    while n1 >= 0 and n2 >= 0:
        if nums1[n1] > nums2[n2]:
            nums1[i] = nums1[n1]
            n1 -= 1
        else:
            nums1[i] = nums2[n2]
            n2 -= 1
        i -= 1
    while n1 >= 0:
        nums1[i] = nums1[n1]
        n1 -= 1
        i -= 1
    while n2 >= 0:
        nums1[i] = nums2[n2]
        n2 -= 1
        i -= 1

# https://leetcode.com/problems/find-the-prefix-common-array-of-two-arrays
def findThePrefixCommonArray(self, A: List[int], B: List[int]) -> List[int]:
    result = [0] * len(A)
    counts = [None] * 51  # better to use len(A) instead of 51
    common = 0
    for i in range(len(A)):
        if not counts[A[i]]:
            counts[A[i]] = 0
        if not counts[B[i]]:
            counts[B[i]] = 0
        counts[A[i]] -= 1
        counts[B[i]] += 1
        if counts[A[i]] == 0:
            common += 1
        if A[i] != B[i] and counts[B[i]] == 0:
            common += 1
        result[i] = common
    return result

# https://leetcode.com/problems/heaters
def findRadius(self, houses: List[int], heaters: List[int]) -> int:
    houses.sort()
    heaters.sort()
    i = 0
    radius = 0
    for house in houses:
        # Move the heater pointer to the nearest heater to the current house
        while i < len(heaters) - 1 and abs(heaters[i + 1] - house) <= abs(heaters[i] - house):
            i += 1
        radius = max(radius, abs(heaters[i] - house))
    return radius
def findRadiusBinarySearch(self, houses: List[int], heaters: List[int]) -> int:
    houses.sort()
    heaters.sort()
    result = 0
    for house in houses:
        idx = bisect.bisect_left(heaters, house)
        # Calculate distances to the closest heaters (left and right)
        left_distance = float('inf') if idx == 0 else abs(house - heaters[idx - 1])
        right_distance = float('inf') if idx == len(heaters) else abs(house - heaters[idx])
        nearest = min(left_distance, right_distance)
        result = max(result, nearest)
    return result

# https://leetcode.com/problems/longest-consecutive-sequence
def longestConsecutive(self, nums: List[int]) -> int:
    lookup = set(nums)
    result = 0
    for n in lookup:
        if n - 1 not in lookup:
            # potential starting point of a subsequence, ensures we don't waste effort checking
            # this subsequence again
            i = n + 1
            while i in lookup:
                i += 1
            result = max(result, i - n)
            if result > len(nums) // 2:
                return result
    return result

# https://leetcode.com/problems/find-minimum-in-rotated-sorted-array
def findMin(self, nums: List[int]) -> int:
    if len(nums) == 1:
        return nums[0]
    lo = 0
    hi = len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid == 0:
            return min(nums[0], nums[1])
        if nums[mid] < nums[mid - 1]:
            return nums[mid]

        if nums[mid] > nums[hi]:
            lo = mid + 1
        else:
            hi = mid - 1
    return lo

# https://leetcode.com/problems/search-in-rotated-sorted-array
def search(self, nums: List[int], target: int) -> int:
    def findMin(nums: List[int]) -> int:
        if len(nums) == 1:
            return 0
        lo = 0
        hi = len(nums) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if mid == 0:
                return 0 if nums[0] < nums[1] else 1
            if nums[mid] < nums[mid - 1]:
                return mid

            if nums[mid] > nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo

    n = len(nums)
    offset = findMin(nums)
    # do regular binary search but when checking numbers check the correct rotated index using offset
    lo = 0
    hi = n - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        cur = (mid + offset) % n
        if nums[cur] == target:
            return cur
        if nums[cur] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
def searchPretty(self, nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid

        if nums[left] <= nums[mid]:  # Left half is sorted
            if nums[left] <= target < nums[mid]:  # Target in left half
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half is sorted
            if nums[mid] < target <= nums[right]:  # Target in right half
                left = mid + 1
            else:
                right = mid - 1
    return -1

# https://leetcode.com/problems/sort-colors
# O(n) time and O(1) space
def sortColors(self, nums: List[int]) -> None:
    lo = -1  # index of last red
    cur = 0  # cur index
    hi = len(nums) - 1  # index before first blue

    while cur <= hi:
        if nums[cur] == 0:  # swap cur red with the index after last red
            lo += 1
            nums[lo], nums[cur] = nums[cur], nums[lo]
            cur += 1
        elif nums[cur] == 1:
            # leave white in the middle
            cur += 1
        else:  # move blue to end
            nums[hi], nums[cur] = nums[cur], nums[hi]
            hi -= 1
            # leave cur where it is since moved unknown value from end to cur

# https://leetcode.com/problems/jump-game
def canJump(self, nums: List[int]) -> bool:
    furthest = 0  # furthest index we can currently jump to
    for i in range(len(nums)):
        furthest = max(furthest, i + nums[i])
        if furthest >= len(nums) - 1:
            return True
        if furthest == i and nums[i] == 0:
            return False
    return False

# https://leetcode.com/problems/maximum-subarray
def maxSubArray(self, nums: List[int]) -> int:
    s = nums[0]  # sum of nums[lo,hi)
    maxSum = nums[0]
    lo = 0
    hi = 1
    while hi < len(nums):
        # a negative sum is worse than starting with nothing
        while s <= 0 and lo < hi:
            s -= nums[lo]
            lo += 1
        s += nums[hi]
        maxSum = max(maxSum, s)
        hi += 1
    return maxSum
def maxSubArrayPretty(self, nums: List[int]) -> int:
    maxSum = nums[0]
    currentSum = nums[0]
    for num in nums[1:]:
        currentSum = max(num, currentSum + num)
        maxSum = max(maxSum, currentSum)
    return maxSum

# https://leetcode.com/problems/find-k-closest-elements
def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
    lo = 0
    hi = len(arr) - k
    # search the window arr[mid]-arr[mid+k]
    while lo < hi:
        mid = lo + (hi-lo) // 2
        if x <= arr[mid]:
            hi = mid
        elif arr[mid+k] <= x:
            lo = mid+1
        else:
            middist = abs(x-arr[mid])
            midkdist = abs(x-arr[mid+k])
            if middist <= midkdist:
                hi = mid  # lo side of window closer to x so move window left
            else:
                lo = mid+1 # hi side closer to x
    return arr[lo:lo+k]
def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
    hi = bisect.bisect_left(arr, x)  # First element >= x
    lo = hi - 1  # Last element < x
    result = []
    # expand a window of the elements closest to x
    while len(result) < k:
        if lo < 0:  # No more elements on the left
            result.append(arr[hi])
            hi += 1
        elif hi >= len(arr):  # No more elements on the right
            result.append(arr[lo])
            lo -= 1
        elif abs(arr[lo] - x) <= abs(arr[hi] - x):  # Pick the closer element
            result.append(arr[lo])
            lo -= 1
        else:
            result.append(arr[hi])
            hi += 1
    return sorted(result)

# https://leetcode.com/problems/product-of-array-except-self
# O(n) time and O(1) space
def productExceptSelf(self, nums: List[int]) -> List[int]:
    result = [1] * len(nums)
    prevProd = 1
    for i in range(len(nums)-2, -1, -1):
        result[i] = result[i+1] * nums[i+1]  # store suffix products
    for i in range(len(nums)):
        result[i] *= prevProd
        prevProd *= nums[i]
    return result

# https://leetcode.com/problems/trapping-rain-water
# O(n) time and O(1) space
def trap(self, height: List[int]) -> int:
    if len(height) < 3:
        return 0
    result = 0
    left, leftMax = 1, height[0]
    right, rightMax = len(height) - 2, height[-1]

    while left <= right:
        level = min(leftMax, rightMax)
        # pick lower side, add the height of the column of water at this point and move to center
        if leftMax < rightMax:
            result += max(0, level - height[left])
            leftMax = max(leftMax, height[left])
            left += 1
        else:
            result += max(0, level - height[right])
            rightMax = max(rightMax, height[right])
            right -= 1
    return result

# https://leetcode.com/problems/contains-duplicate-ii
def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
    valToIndex = {} # val -> most recent index of val
    for i, n in enumerate(nums):
        if n in valToIndex:
            if i - valToIndex[n] <= k:
                return True
        valToIndex[n] = i
    return False

# https://leetcode.com/problems/remove-duplicates-from-sorted-array
def removeDuplicates(self, nums: List[int]) -> int:
    i = 1
    for j in range(1, len(nums)):
        if nums[j] != nums[j-1]:
            nums[i] = nums[j]
            i += 1
    return i

# https://leetcode.com/problems/continuous-subarray-sum
def checkSubarraySum(self, nums: List[int], k: int) -> bool:
    # compute the prefix mods (mod of the sum before each index)
    # save the first index for each mod in a dict
    # if a later prefix mod is the same, then the sum of the subarray between % k must be 0
    firstMod = {0:-1} # modValue -> index of first occurrence of it
    s = 0
    for i, n in enumerate(nums):
        s += n
        mod = s % k
        if mod not in firstMod:
            firstMod[mod] = i
        elif i - firstMod[mod] > 1:
            return True
    return False
def checkSubarraySumBrute(self, nums: List[int], k: int) -> bool:
    for i in range(len(nums)-1):
        curSum = nums[i]
        for j in range(i+1, len(nums)):
            curSum += nums[j]
            if curSum % k == 0:
                return True
    return False

# https://leetcode.com/problems/move-zeroes
def moveZeroes(self, nums: List[int]) -> None:
    try:
        first0 = nums.index(0)
    except:
        return

    i = first0 + 1
    while i < len(nums):
        if nums[i] != 0:
            nums[first0], nums[i] = nums[i], 0
            while first0 <= i and nums[first0] != 0:
                first0 += 1
        i += 1

# https://leetcode.com/problems/best-time-to-buy-and-sell-stock
def maxProfit(self, prices: List[int]) -> int:
    result = 0
    maxBefore = prices[-1]
    for i in range(len(prices)-2, -1, -1):
        result = max(result, maxBefore - prices[i])
        maxBefore = max(maxBefore, prices[i])
    return result

# https://leetcode.com/problems/kth-missing-positive-number
def findKthPositive(self, arr: List[int], k: int) -> int:
    missing = arr[0] - 1
    if missing >= k:
        return k
    for i in range(1, len(arr)):
        gap = arr[i] - arr[i - 1] - 1
        missing += gap
        if missing >= k:
            return arr[i] - 1 - (missing - k)
    return arr[-1] + (k - missing)
def findKthPositive2(self, A, k):
    l, r = 0, len(A)
    while l < r:
        m = (l + r) / 2
        if A[m] - 1 - m < k:
            l = m + 1
        else:
            r = m
    return l + k

# https://leetcode.com/problems/next-permutation
'''
First, we observe that for any given sequence that is in descending order, no next larger permutation is possible.
For example, no next permutation is possible for the following array:
We need to find the first pair of two successive numbers a[i] and a[i−1], from the right, which satisfy
a[i]>a[i−1]. Now, no rearrangements to the right of a[i−1] can create a larger permutation since that subarray consists of numbers in descending order.
Thus, we need to rearrange the numbers to the right of a[i−1] including itself.
Now, what kind of rearrangement will produce the next larger number? We want to create the permutation just larger than the current one. Therefore, we need to replace the number a[i−1] with the number which is just larger than itself among the numbers lying to its right section, say a[j].
We swap the numbers a[i−1] and a[j]. We now have the correct number at index i−1. But still the current permutation isn't the permutation
that we are looking for. We need the smallest permutation that can be formed by using the numbers only to the right of a[i−1]. Therefore, we need to place those
numbers in ascending order to get their smallest permutation.
But, recall that while scanning the numbers from the right, we simply kept decrementing the index
until we found the pair a[i] and a[i−1] where, a[i]>a[i−1]. Thus, all numbers to the right of a[i−1] were already sorted in descending order.
Furthermore, swapping a[i−1] and a[j] didn't change that order.
Therefore, we simply need to reverse the numbers following a[i−1] to get the next smallest lexicographic permutation.
'''
# O(n) time and O(1) space
def nextPermutation(self, nums):
    i = len(nums) - 2
    while i >= 0 and nums[i + 1] <= nums[i]:
        i -= 1
    if i >= 0:
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        self.swap(nums, i, j)
    self.reverse(nums, i + 1)
def reverse(self, nums, start):
    i, j = start, len(nums) - 1
    while i < j:
        self.swap(nums, i, j)
        i += 1
        j -= 1
def swap(self, nums, i, j):
    temp = nums[i]
    nums[i] = nums[j]
    nums[j] = temp
def nextPermutationOG(self, nums: List[int]) -> None:
    # observe: if the 2nd to last is smaller than the last, swap last 2 numbers, otherwise:
    # continue moving forward in number from the end until you find a number smaller than the num to its right

    if len(nums) <= 1:
        return nums

    beforeHi = len(nums) - 2
    # if equal, grab the leftmost number and move to end
    while beforeHi >= 0 and nums[beforeHi] >= nums[beforeHi + 1]:
        beforeHi -= 1

    if beforeHi == -1:
        nums.sort()
    else:
        # for the numbers to the right of beforeHi we must swap the smallest number larger than nums[beforeHi] at beforeHi and then sort the numbers to the right of beforeHi
        minHigherI = beforeHi + 1
        for i in range(beforeHi + 2, len(nums)):
            if nums[i] > nums[beforeHi] and nums[i] < nums[minHigherI]:
                minHigherI = i

        temp = nums[beforeHi]
        nums[beforeHi] = nums[minHigherI]
        nums[minHigherI] = temp

        nums[beforeHi + 1:] = sorted(nums[beforeHi + 1:])

# https://leetcode.com/problems/max-consecutive-ones-iii
def longestOnes(self, nums: List[int], k: int) -> int:
    lo = 0
    hi = 0
    zeros = 0
    result = 0
    while hi < len(nums):
        if nums[hi] == 0:
            zeros += 1
        while zeros > k:
            if nums[lo] == 0:
                zeros -= 1
            lo += 1
        result = max(result, hi-lo+1)
        hi += 1
    return result

# https://leetcode.com/problems/increasing-triplet-subsequence
def increasingTriplet(self, nums: List[int]) -> bool:
    lo = float('inf')  # tracks the min value in nums
    mid = float('inf')  # mid gets set if there is a valid lower number before it
    for n in nums:
        if n <= lo:
            lo = n
        elif n <= mid:
            mid = n
        else:
            return True
    return False

# https://leetcode.com/problems/koko-eating-bananas
# O(n*logm) time
def minEatingSpeed(self, piles: List[int], h: int) -> int:
    lo = 1 # smallest possible k value
    hi = max(piles) # largest possible k value
    while lo <= hi:
        mid = (lo + hi) // 2
        hour_spent = 0 # test using mid as the bite size k
        for pile in piles:
            hour_spent += math.ceil(pile / mid)
        if hour_spent <= h:
            hi = mid - 1
        else:
            lo = mid + 1
    return lo  # finished eating all bananas

# https://leetcode.com/problems/search-suggestions-system
# O(nlog(n))+O(mlog(n)) time
def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
    products.sort()
    result = []
    search = ''
    for end in range(len(searchWord)):
        search += searchWord[end]
        r = []
        target = bisect_left(products, search)
        for i in range(target, min(target + 3, len(products))):
            if products[i].startswith(search):
                r.append(products[i])
            else:
                break
        result.append(r)
    return result

# https://leetcode.com/problems/last-stone-weight
#counting sort also an option
def lastStoneWeight(self, stones: List[int]) -> int:
    stones = [-s for s in stones]
    heapq.heapify(stones)
    while len(stones) > 1:
        a = heapq.heappop(stones)
        b = heapq.heappop(stones)
        if a != b:
            heapq.heappush(stones, -(-a - -b))
    return -stones[0] if stones else 0

# https://leetcode.com/problems/number-of-visible-people-in-a-queue/
'''
[10,6,8,5,11,9] -> []
[10,6,8,5,11,9] -> [9]
[10,6,8,5,11,9] -> [11]
[10,6,8,5,11,9] -> [5,11]
[10,6,8,5,11,9] -> [8,11] -> actually can't see 11 since 8 is > 6
[10,6,8,5,11,9] -> [6,8,11]
'''
def canSeePersonsCount(self, heights: List[int]) -> List[int]:
    result = [0] * len(heights)
    q = deque()  # monotonic increasing heights of people to our right
    for i in range(len(heights) - 1, -1, -1):
        cur = heights[i]
        shorter = 0
        for o in q: # could binary search in sorted q instead of linear
            if o < cur:
                shorter += 1
            else:
                break
        if shorter < len(q):
            shorter += 1  # can see one person taller than us beyond the people shorter than us
        # print('cur', cur, 'q', q, 'shorter', shorter)
        result[i] = shorter
        while q and q[0] < cur: # runs a total of n iterations
            q.popleft()
        q.appendleft(cur)
    return result
def canSeePersonsCountBest(self, A):
    res = [0] * len(A)
    stack = [] # indexes of monotonic decreasing heights
    for i, v in enumerate(A):
        # print('cur', v, 'stack', stack)
        while stack and A[stack[-1]] <= v: # smaller top of stack <= current val, they can see current val but not beyond
            res[stack.pop()] += 1
        if stack:
            res[stack[-1]] += 1 # current val < top of stack so they can see us
        stack.append(i) # store index of current element
    return res

# https://leetcode.com/problems/longest-strictly-increasing-or-strictly-decreasing-subarray
def longestMonotonicSubarray(self, nums: List[int]) -> int:
    result = 1
    curLen = 1
    incr = False
    prev = nums[0]
    for i in range(1, len(nums)):
        if nums[i] > prev:
            if incr != True:  # at the 2nd element in increasing subarray
                incr = True
                curLen = 2
            else:
                curLen += 1
        elif nums[i] < prev:
            if incr != False:  # at the 2nd element in decreasing subarray
                incr = False
                curLen = 2
            else:
                curLen += 1
        else:
            incr = None
        prev = nums[i]
        result = max(result, curLen)
    return result

# https://leetcode.com/problems/check-if-array-is-sorted-and-rotated
def check(self, nums: List[int]) -> bool:
    seenDrop = nums[-1] > nums[0]
    for i in range(len(nums) - 1):
        if nums[i + 1] < nums[i]:
            if seenDrop:
                return False
            seenDrop = True
    return True

# https://leetcode.com/problems/two-sum-ii-input-array-is-sorted
def twoSum(self, numbers: List[int], target: int) -> List[int]:
    lo = 0
    hi = len(numbers) - 1
    while lo < hi:
        s = numbers[lo] + numbers[hi]
        if s == target:
            return [lo + 1, hi + 1]
        if s < target:
            lo += 1
        else:
            hi -= 1
    return None

# https://leetcode.com/problems/missing-ranges
def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[List[int]]:
    if not nums:
        return [[lower, upper]]
    missing = []
    if lower < nums[0]:
        missing.append([lower, nums[0] - 1])

    for i in range(1, len(nums)):
        if nums[i - 1] != nums[i] and nums[i - 1] != nums[i] - 1:
            missing.append([nums[i - 1] + 1, nums[i] - 1])

    if nums[-1] < upper:
        missing.append([nums[-1] + 1, upper])
    return missing

# https://leetcode.com/problems/max-consecutive-ones
def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
    ones = 0
    maxOnes = 0
    for n in nums:
        if n == 1:
            ones += 1
            maxOnes = max(maxOnes, ones)
        else:
            ones = 0
    return maxOnes

# https://leetcode.com/problems/special-array-i
def isArraySpecial(self, nums: List[int]) -> bool:
    for i in range(1, len(nums)):
        if nums[i - 1] % 2 == nums[i] % 2:
            return False
    return True

# https://leetcode.com/problems/range-sum-query-immutable
class NumArray:
    def __init__(self, nums: List[int]):
        self.prefixSums = [0] * len(nums)
        s = 0
        for i in range(len(nums)):
            s += nums[i]
            self.prefixSums[i] = s

    # 1,2,3,4
    # 1,3,6,10
    def sumRange(self, left: int, right: int) -> int:
        if left - 1 >= 0:
            return self.prefixSums[right] - self.prefixSums[left - 1]
        return self.prefixSums[right]

# https://leetcode.com/problems/house-robber-ii
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 0 or nums is None:
            return 0
        if len(nums) == 1:
            return nums[0]
        return max(self.rob_simple(nums[:-1]), self.rob_simple(nums[1:]))

    def rob_simple(self, nums: List[int]) -> int:
        t1 = 0
        t2 = 0
        for current in nums:
            temp = t1
            t1 = max(current + t2, t1)
            t2 = temp
        return t1

# https://leetcode.com/problems/rotate-array
def rotate(self, nums: List[int], k: int) -> None:
    n = len(nums)
    k %= n

    start = count = 0
    while count < n:
        current, prev = start, nums[start]
        while True:
            next_idx = (current + k) % n
            nums[next_idx], prev = prev, nums[next_idx]
            current = next_idx
            count += 1

            if start == current:
                break
        start += 1

# https://leetcode.com/problems/first-missing-positive
# replace all 0 and negative values with 1
# O(n) time and space
def firstMissingPositive(self, nums: List[int]) -> int:
    n = len(nums)
    contains_1 = False

    # Replace negative numbers, zeros, and numbers larger than n with 1s.
    # After this nums contains only positive numbers.
    for i in range(n):
        # Check whether 1 is in the original array
        if nums[i] == 1:
            contains_1 = True
        if nums[i] <= 0 or nums[i] > n:
            nums[i] = 1
    if not contains_1:
        return 1

    # Mark whether integers 1 to n are in nums
    # Use index as a hash key and negative sign as a presence detector.
    for i in range(n):
        value = abs(nums[i])
        # If you meet number a in the array - change the sign of a-th element.
        # Be careful with duplicates : do it only once.
        if value == n:
            nums[0] = -abs(nums[0])
        else:
            nums[value] = -abs(nums[value])
    # First positive in nums is smallest missing positive integer
    for i in range(1, n):
        if nums[i] > 0:
            return i
    # nums[0] stores whether n is in nums
    if nums[0] > 0:
        return n
    # If nums contained all elements 1 to n
    # the smallest missing positive number is n + 1
    return n + 1
# Cycle sort is a sorting algorithm that can sort a given sequence in a range from a to n by putting each element at the index that corresponds to its value.
def firstMissingPositive2(self, nums: List[int]) -> int:
    n = len(nums)
    # Use cycle sort to place positive elements smaller than n at the correct index
    i = 0
    while i < n:
        correct_idx = nums[i] - 1
        if 0 < nums[i] <= n and nums[i] != nums[correct_idx]:
            # swap
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    # Iterate through nums return smallest missing positive integer
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    # If all elements are at the correct index
    # the smallest missing positive number is n + 1
    return n + 1

# https://leetcode.com/problems/summary-ranges
def summaryRanges(self, nums: List[int]) -> List[str]:
    ranges = []
    i = 0
    while i < len(nums):
        start = nums[i]
        while i + 1 < len(nums) and nums[i] + 1 == nums[i + 1]:
            i += 1
        if start != nums[i]:
            ranges.append(str(start) + "->" + str(nums[i]))
        else:
            ranges.append(str(nums[i]))
        i += 1
    return ranges

# https://leetcode.com/problems/coin-change
# O(s*n) time and O(s) space
def coinChange(self, coins: List[int], amount: int) -> int:
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

# https://leetcode.com/problems/sliding-window-median
# vector<double> medianSlidingWindow(vector<int>& nums, int k)
# {
#     vector<double> medians;
#     multiset<int> lo, hi;
#
#     for (int i = 0; i < nums.size(); i++) {
#         //remove outgoing element
#         if (i >= k) {
#             if (nums[i - k] <= *lo.rbegin())
#                 lo.erase(lo.find(nums[i - k]));
#             else
#                 hi.erase(hi.find(nums[i - k]));
#         }
#
#         // insert incoming element
#         lo.insert(nums[i]);
#
#         // balance the sets
#         hi.insert(*lo.rbegin());
#         lo.erase(prev(lo.end()));
#
#         if (lo.size() < hi.size()) {
#             lo.insert(*hi.begin());
#             hi.erase(hi.begin());
#         }
#
#         // get median
#         if (i >= k - 1) {
#             medians.push_back(k & 1 ? *lo.rbegin() : ((double)(*lo.rbegin()) + (double)(*hi.begin())) * 0.5);
#         }
#     }
#     return medians;
# }

# https://leetcode.com/problems/median-of-two-sorted-arrays/
# time: O(log(m⋅n)), space: O(logm+logn)
def findMedianSortedArrays(self, A: List[int], B: List[int]) -> float:
    na, nb = len(A), len(B)
    n = na + nb

    def solve(k, a_start, a_end, b_start, b_end):
        # If the segment of on array is empty, it means we have passed all
        # its element, just return the corresponding element in the other array.
        if a_start > a_end:
            return B[k - a_start]
        if b_start > b_end:
            return A[k - b_start]

        # Get the middle indexes and middle values of A and B.
        a_index, b_index = (a_start + a_end) // 2, (b_start + b_end) // 2
        a_value, b_value = A[a_index], B[b_index]

        # If k is in the right half of A + B, remove the smaller left half.
        if a_index + b_index < k:
            if a_value > b_value:
                return solve(k, a_start, a_end, b_index + 1, b_end)
            else:
                return solve(k, a_index + 1, a_end, b_start, b_end)
        # Otherwise, remove the larger right half.
        else:
            if a_value > b_value:
                return solve(k, a_start, a_index - 1, b_start, b_end)
            else:
                return solve(k, a_start, a_end, b_start, b_index - 1)

    if n % 2:
        return solve(n // 2, 0, na - 1, 0, nb - 1)
    else:
        return (
            solve(n // 2 - 1, 0, na - 1, 0, nb - 1)
            + solve(n // 2, 0, na - 1, 0, nb - 1)
        ) / 2
# Time complexity: O(log(min(m,n))), space O(1)
def findMedianSortedArrays(
    self, nums1: List[int], nums2: List[int]
) -> float:
    if len(nums1) > len(nums2):
        return self.findMedianSortedArrays(nums2, nums1)

    m, n = len(nums1), len(nums2)
    left, right = 0, m

    while left <= right:
        partitionA = (left + right) // 2
        partitionB = (m + n + 1) // 2 - partitionA

        maxLeftA = (
            float("-inf") if partitionA == 0 else nums1[partitionA - 1]
        )
        minRightA = float("inf") if partitionA == m else nums1[partitionA]
        maxLeftB = (
            float("-inf") if partitionB == 0 else nums2[partitionB - 1]
        )
        minRightB = float("inf") if partitionB == n else nums2[partitionB]

        if maxLeftA <= minRightB and maxLeftB <= minRightA:
            if (m + n) % 2 == 0:
                return (
                    max(maxLeftA, maxLeftB) + min(minRightA, minRightB)
                ) / 2
            else:
                return max(maxLeftA, maxLeftB)
        elif maxLeftA > minRightB:
            right = partitionA - 1
        else:
            left = partitionA + 1

# https://leetcode.com/problems/stickers-to-spell-word
# O(2^t *s*t) time and O(2^t) space
def minStickers(self, stickers, target):
    t_count = collections.Counter(target)
    A = [collections.Counter(sticker) & t_count
         for sticker in stickers]

    for i in range(len(A) - 1, -1, -1):
        if any(A[i] == A[i] & A[j] for j in range(len(A)) if i != j):
            A.pop(i)

    stickers = ["".join(s_count.elements()) for s_count in A]
    dp = [-1] * (1 << len(target))
    dp[0] = 0
    for state in range(1 << len(target)):
        if dp[state] == -1: continue
        for sticker in stickers:
            now = state
            for letter in sticker:
                for i, c in enumerate(target):
                    if (now >> i) & 1: continue
                    if c == letter:
                        now |= 1 << i
                        break
            if dp[now] == -1 or dp[now] > dp[state] + 1:
                dp[now] = dp[state] + 1
    return dp[-1]
def minStickersExhaustiveSearch(self, stickers, target):
    t_count = collections.Counter(target)
    A = [collections.Counter(sticker) & t_count
         for sticker in stickers]
    for i in range(len(A) - 1, -1, -1):
        if any(A[i] == A[i] & A[j] for j in range(len(A)) if i != j):
            A.pop(i)

    self.best = len(target) + 1
    def search(ans = 0):
        if ans >= self.best: return
        if not A:
            if all(t_count[letter] <= 0 for letter in t_count):
                self.best = ans
            return

        sticker = A.pop()
        used = max((t_count[letter] - 1) // sticker[letter] + 1
                    for letter in sticker)
        used = max(used, 0)

        for c in sticker:
            t_count[c] -= used * sticker[c]

        search(ans + used)
        for i in range(used - 1, -1, -1):
            for letter in sticker:
                t_count[letter] += sticker[letter]
            search(ans + i)

        A.append(sticker)
    search()
    return self.best if self.best <= len(target) else -1

# https://leetcode.com/problems/count-of-smaller-numbers-after-self
# segment tree, time = O(nlogm) space = O(m)
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        # implement segment tree
        def update(index, value, tree, size):
            index += size  # shift the index to the leaf
            # update from leaf to root
            tree[index] += value
            while index > 1:
                index //= 2
                tree[index] = tree[index * 2] + tree[index * 2 + 1]

        def query(left, right, tree, size):
            # return sum of [left, right)
            result = 0
            left += size  # shift the index to the leaf
            right += size
            while left < right:
                # if left is a right node
                # bring the value and move to parent's right node
                if left % 2 == 1:
                    result += tree[left]
                    left += 1
                # else directly move to parent
                left //= 2
                # if right is a right node
                # bring the value of the left node and move to parent
                if right % 2 == 1:
                    right -= 1
                    result += tree[right]
                # else directly move to parent
                right //= 2
            return result

        offset = 10**4  # offset negative to non-negative
        size = 2 * 10**4 + 1  # total possible values in nums
        tree = [0] * (2 * size)
        result = []
        for num in reversed(nums):
            smaller_count = query(0, num + offset, tree, size)
            result.append(smaller_count)
            update(num + offset, 1, tree, size)
        return reversed(result)
# fenwick tree, time = O(nlogm) space = O(m)
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        # implement Binary Index Tree
        def update(index, value, tree, size):
            index += 1  # index in BIT is 1 more than the original index
            while index < size:
                tree[index] += value
                index += index & -index

        def query(index, tree):
            # return sum of [0, index)
            result = 0
            while index >= 1:
                result += tree[index]
                index -= index & -index
            return result

        offset = 10**4  # offset negative to non-negative
        size = 2 * 10**4 + 2  # total possible values in nums plus one dummy
        tree = [0] * size
        result = []
        for num in reversed(nums):
            smaller_count = query(num + offset, tree)
            result.append(smaller_count)
            update(num + offset, 1, tree, size)
        return reversed(result)
# merge sort O(nlogn) time and O(n) space
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        n = len(nums)
        arr = [[v, i] for i, v in enumerate(nums)]  # record value and index
        result = [0] * n

        def merge_sort(arr, left, right):
            # merge sort [left, right) from small to large, in place
            if right - left <= 1:
                return
            mid = (left + right) // 2
            merge_sort(arr, left, mid)
            merge_sort(arr, mid, right)
            merge(arr, left, right, mid)

        def merge(arr, left, right, mid):
            # merge [left, mid) and [mid, right)
            i = left  # current index for the left array
            j = mid  # current index for the right array
            # use temp to temporarily store sorted array
            temp = []
            while i < mid and j < right:
                if arr[i][0] <= arr[j][0]:
                    # j - mid numbers jump to the left side of arr[i]
                    result[arr[i][1]] += j - mid
                    temp.append(arr[i])
                    i += 1
                else:
                    temp.append(arr[j])
                    j += 1
            # when one of the subarrays is empty
            while i < mid:
                # j - mid numbers jump to the left side of arr[i]
                result[arr[i][1]] += j - mid
                temp.append(arr[i])
                i += 1
            while j < right:
                temp.append(arr[j])
                j += 1
            # restore from temp
            for i in range(left, right):
                arr[i] = temp[i - left]

        merge_sort(arr, 0, n)

        return result

# https://leetcode.com/problems/candy
# O(n) time and space
def candy(self, ratings: List[int]) -> int:
    sum = 0
    n = len(ratings)
    left2right = [1] * n
    right2left = [1] * n
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            left2right[i] = left2right[i - 1] + 1
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            right2left[i] = right2left[i + 1] + 1
    for i in range(n):
        sum += max(left2right[i], right2left[i])
    return sum
def candy(self, ratings):
    candies = [1] * len(ratings)
    for i in range(1, len(ratings)):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1
    sum = candies[-1]
    for i in range(len(ratings) - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)
        sum += candies[i]
    return sum
# Function to calculate sum of first n natural numbers
def count(self, n):
    return (n * (n + 1)) // 2
# O(n) time and O(1) space
def candy(self, ratings):
    if len(ratings) <= 1:
        return len(ratings)
    candies = 0
    up = 0
    down = 0
    oldSlope = 0
    for i in range(1, len(ratings)):
        newSlope = (
            1
            if ratings[i] > ratings[i - 1]
            else (-1 if ratings[i] < ratings[i - 1] else 0)
        )
        # slope is changing from uphill to flat or downhill
        # or from downhill to flat or uphill
        if (oldSlope > 0 and newSlope == 0) or (
            oldSlope < 0 and newSlope >= 0
        ):
            candies += self.count(up) + self.count(down) + max(up, down)
            up = 0
            down = 0
        # slope is uphill
        if newSlope > 0:
            up += 1
        # slope is downhill
        elif newSlope < 0:
            down += 1
        # slope is flat
        else:
            candies += 1
        oldSlope = newSlope
    candies += self.count(up) + self.count(down) + max(up, down) + 1
    return candies

# https://leetcode.com/problems/utf-8-validation
# O(n) time and space
def validUtf8(self, data):
    """
    :type data: List[int]
    :rtype: bool
    """

    # Number of bytes in the current UTF-8 character
    n_bytes = 0

    # For each integer in the data array.
    for num in data:

        # Get the binary representation. We only need the least significant 8 bits
        # for any given number.
        bin_rep = format(num, '#010b')[-8:]

        # If this is the case then we are to start processing a new UTF-8 character.
        if n_bytes == 0:

            # Get the number of 1s in the beginning of the string.
            for bit in bin_rep:
                if bit == '0': break
                n_bytes += 1

            # 1 byte characters
            if n_bytes == 0:
                continue

            # Invalid scenarios according to the rules of the problem.
            if n_bytes == 1 or n_bytes > 4:
                return False
        else:
            # Else, we are processing integers which represent bytes which are a part of
            # a UTF-8 character. So, they must adhere to the pattern `10xxxxxx`.
            if not (bin_rep[0] == '1' and bin_rep[1] == '0'):
                return False

        # We reduce the number of bytes to process by 1 after each integer.
        n_bytes -= 1

    # This is for the case where we might not have the complete data for
    # a particular UTF-8 character.
    return n_bytes == 0
# O(n) time and O(1) space
def validUtf8(self, data):
    """
    :type data: List[int]
    :rtype: bool
    """

    # Number of bytes in the current UTF-8 character
    n_bytes = 0

    # Mask to check if the most significant bit (8th bit from the left) is set or not
    mask1 = 1 << 7

    # Mask to check if the second most significant bit is set or not
    mask2 = 1 << 6
    for num in data:

        # Get the number of set most significant bits in the byte if
        # this is the starting byte of an UTF-8 character.
        mask = 1 << 7
        if n_bytes == 0:
            while mask & num:
                n_bytes += 1
                mask = mask >> 1

            # 1 byte characters
            if n_bytes == 0:
                continue

            # Invalid scenarios according to the rules of the problem.
            if n_bytes == 1 or n_bytes > 4:
                return False
        else:

            # If this byte is a part of an existing UTF-8 character, then we
            # simply have to look at the two most significant bits and we make
            # use of the masks we defined before.
            if not (num & mask1 and not (num & mask2)):
                return False
        n_bytes -= 1
    return n_bytes == 0