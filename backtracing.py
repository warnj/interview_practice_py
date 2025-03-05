from typing import List

# https://leetcode.com/problems/subsets/solutions/27281/a-general-approach-to-backtracking-questions-in-java-subsets-permutations-combination-sum-palindrome-partitioning/

# https://leetcode.com/problems/permutations
# Time complexity, what you should say in an interview: O(n⋅n!)  Space complexity: O(n)
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
# time = n * 2^n
# space = n
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
# time = n * 2^n
# space = n
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
# Time complexity: O(n⋅2^n), space = n
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
# Time complexity: O(n⋅2^n), space = log n
def subsetsWithDupItr(self, nums):
    nums.sort()
    subsets = [[]]
    subsetSize = 0
    for i in range(len(nums)):
        # subsetSize refers to the size of the subset in the previous step.
        # This value also indicates the starting index of the subsets generated in this step.
        startingIndex = (
            subsetSize if i >= 1 and nums[i] == nums[i - 1] else 0
        )
        subsetSize = len(subsets)
        for j in range(startingIndex, subsetSize):
            currentSubset = list(subsets[j])
            currentSubset.append(nums[i])
            subsets.append(currentSubset)
    return subsets

# all subsequences: A subsequence is a string that can be derived from another string by deleting some or no characters without changing the order of the remaining characters.
def printSubsequences(self, s: str) -> int:
    def explore(i, subs):
        if i == len(s):
            print(''.join(subs))
        else:
            # keep the current char
            explore(i + 1, subs)
            # delete the current char
            temp = subs[i]
            subs[i] = ''
            explore(i + 1, subs)
            subs[i] = temp

    explore(0, list(s))
    return -1

# time O(4^n / sqrt(n)), space O(n)
def generateParenthesis(self, n: int) -> List[str]:
    answer = []

    def backtracking(cur_string, left_count, right_count):
        if len(cur_string) == 2 * n:
            answer.append("".join(cur_string))
            return
        if left_count < n:
            cur_string.append("(")
            backtracking(cur_string, left_count + 1, right_count)
            cur_string.pop()
        if right_count < left_count:
            cur_string.append(")")
            backtracking(cur_string, left_count, right_count + 1)
            cur_string.pop()

    backtracking([], 0, 0)
    return answer

def generateParenthesis(self, n: int) -> List[str]:
    out = []
    def helper(br: str, left: int, right: int):
        if len(br) == 2 * n:
            out.append(br)
        if left > 0:
            helper(br + '(', left - 1, right)
        if right > 0 and right > left:
            helper(br + ')', left, right - 1)

    helper("", n, n)
    return out
