from typing import List
import random

def quicksort(nums: List[int]) -> None:
    return _quicksort(nums, 0, len(nums)-1)

def _quicksort(nums, lo, hi):
    if hi - lo > 0:
        correctlyOrderedElemIndex = _partition(nums, lo, hi)
        _quicksort(nums, lo, correctlyOrderedElemIndex-1)
        _quicksort(nums, correctlyOrderedElemIndex+1, hi)

def _partition(nums, lo, hi):
    # if you prefer to randomly select pivot instead of using hi you can swap random val with nums[hi]:
    # pivotIndex = random.randint(lo, hi)
    # nums[pivotIndex], nums[hi] = nums[hi], nums[pivotIndex]

    pivot = nums[hi]
    i = lo - 1 # upper end of the range (inclusive) smaller than pivot val
    for j in range(lo, hi):
        if nums[j] < pivot:
            # put elements < nums[p] to the left and >= to the right
            i += 1
            nums[i], nums[j] = nums[j], nums[i]
    nums[i+1], nums[hi] = nums[hi], nums[i+1]
    return i + 1  # index of the correctly located element

def quicksort2(nums: List[int]) -> None:
    def quickSort(lo, hi):
        if hi - lo > 0:
            l, r = lo, hi
            m = l + (r - l) // 2
            pivot = nums[m]
            # find element left of pivot that's >= it and element right of pivot smaller than it, swap
            while r >= l:
                while r >= l and nums[l] < pivot: l += 1
                while r >= l and nums[r] > pivot: r -= 1
                if r >= l:
                    nums[l], nums[r] = nums[r], nums[l]
                    l += 1
                    r -= 1
            quickSort(lo, r)
            quickSort(l, hi)
    quickSort(0, len(nums) - 1)
