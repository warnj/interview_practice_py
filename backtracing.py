from typing import List

# https://leetcode.com/problems/subsets/solutions/27281/a-general-approach-to-backtracking-questions-in-java-subsets-permutations-combination-sum-palindrome-partitioning/

# https://leetcode.com/problems/permutations
def permute(self, nums: List[int]) -> List[List[int]]:
    result = []
    def explore(curList):
        if len(curList) == len(nums):
            result.append(list(curList))
        else:
            for i in range(0, len(nums)):
                if nums[i] in curList:
                    continue
                # print(f'{curList} + {nums[i]}')
                curList.append(nums[i])
                explore(curList)
                curList.pop()
    explore([])
    return result

# https://leetcode.com/problems/subsets
def subsetsIterative(self, nums: List[int]) -> List[List[int]]:
    # [[],[1]] -> 2 = 2^1
    # [[],[1],[2],[1,2]] -> 4 = 2^2
    # [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]] -> 8 = 2^3
    # [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3],[4],[1,4],[2,4],[3,4],[1,2,4],[1,3,4],[2,3,4],[1,2,3,4]] -> 16 = 2^4
    result = []
    for i in range(pow(2, len(nums))):
        s = bin(i)[2:]
        subset = []
        lo = 0
        for j in range(len(s)-1, -1, -1): # no leading zeros so work r to l on number
            if s[j] == '1':
                subset.append(nums[lo])  # work l to r in the list
            lo += 1
        result.append(subset)
    return result
def subsets(self, nums: List[int]) -> List[List[int]]:
    result = []
    def recurse(curList, start):
        result.append(list(curList))
        for i in range(start, len(nums)):
            curList.append(nums[i])
            recurse(curList, i+1)
            curList.pop()
    recurse([], 0)
    return result

# https://leetcode.com/problems/subsets-ii
def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
    result = []
    nums.sort()
    def recurse(curList, start):
        result.append(list(curList))
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i-1]:
                continue
            curList.append(nums[i])
            recurse(curList, i+1)
            curList.pop()
    recurse([], 0)
    return result
