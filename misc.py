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

'''
class Solution {
    private int[] nums;
    private Random rand;
    
    public Solution(int[] nums) {
        this.nums = nums;
        this.rand = new Random();
    }
    
    public int pick(int target) {
        int n = this.nums.length;
        int count = 0;
        int idx = 0;
        for (int i = 0; i < n; ++i) {
            // if nums[i] is equal to target, i is a potential candidate
            // which needs to be chosen uniformly at random
            if (this.nums[i] == target) {
                // increment the count of total candidates
                // available to be chosen uniformly at random
                count++;
                // we pick the current number with probability 1 / count (reservoir sampling)
                if (rand.nextInt(count) == 0) {
                    idx = i;
                }
            }
        }
        return idx;
    }
}
'''