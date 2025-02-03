import random
from typing import List

# https://leetcode.com/problems/random-pick-index
class RandomIndex:
    def __init__(self, nums: List[int]):
        self.vals = {}  # val -> [indexes of val]
        for i, n in enumerate(nums):
            if n in self.vals:
                self.vals[n].append(i)
            else:
                self.vals[n] = [i]
            # self.vals.get(n, []).append(i)

    def pick(self, target: int) -> int:
        return random.choice(self.vals[target])
