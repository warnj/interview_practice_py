import bisect
import heapq
import random
from typing import List

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
def findPeakElement(self, nums: List[int]) -> int:
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

# https://leetcode.com/problems/subarray-sum-equals-k
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
# O(n) time and space
def subarraySum(self, nums: List[int], k: int) -> int:
    count = 0
    total_sum = 0
    prefix_sum_map = {0: 1}
    for num in nums:
        total_sum += num
        if total_sum - k in prefix_sum_map:
            count += prefix_sum_map[total_sum - k]
        prefix_sum_map[total_sum] = prefix_sum_map.get(total_sum, 0) + 1
    return count

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
    firstMod = {0:-1} # modValue -> index of first occurance of it
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
def nextPermutation(self, nums: List[int]) -> None:
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

