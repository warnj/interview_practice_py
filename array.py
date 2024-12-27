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
