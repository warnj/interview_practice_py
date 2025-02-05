from typing import List


# https://leetcode.com/problems/binary-search
def binarySearch(elems, target):
    left = 0
    right = len(elems) - 1
    while left <= right:
        mid = (left + right) // 2
        if elems[mid] == target:
            return mid
        if elems[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array
def searchRange(self, nums: List[int], target: int) -> List[int]:
    result = [-1, -1]
    # binary search for start
    lo = 0
    hi = len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target and (mid - 1 < 0 or nums[mid - 1] < target):
            result[0] = mid
            break
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    # binary search for end
    lo = 0
    hi = len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        # print(f'lo {lo} mid {mid} hi {hi}')
        if nums[mid] == target and (mid + 1 >= len(nums) or nums[mid + 1] > target):
            result[1] = mid
            break
        elif nums[mid] <= target:  # need to move up if equal to target
            lo = mid + 1
        else:
            hi = mid - 1
    return result
def searchRangePretty(self, nums: List[int], target: int) -> List[int]:
    def search(x):
        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if nums[mid] < x:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo
    lo = search(target)
    hi = search(target + 1) - 1
    if lo <= hi:
        return [lo, hi]
    return [-1, -1]

# https://leetcode.com/problems/single-element-in-a-sorted-array
def singleNonDuplicate(self, nums: List[int]) -> int:
    if len(nums) == 1 or nums[1] != nums[0]:
        return nums[0]
    lo = 0
    hi = len(nums) - 1
    while lo <= hi:
        # if first number of a pair is at an even index (0, 2, 4 ...) then have not seen the solo number yet
        mid = (lo + hi) // 2
        pre = mid - 1

        if nums[pre] != nums[mid]:
            # found target
            if mid + 1 == len(nums) or nums[mid + 1] != nums[mid]:
                return nums[mid]
            # mid is the first number of a pair
            if mid % 2 == 0:
                lo = mid + 1
            else:
                hi = mid - 1
        else:
            # pre is the first number of a pair
            if pre % 2 == 0:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
