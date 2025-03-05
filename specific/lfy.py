# https://leetcode.com/problems/lfu-cache
from collections import OrderedDict

class LFUCache:
    # O(n) space
    def __init__(self, capacity: int):
        self.capacity = capacity # must be > 0
        self.minFreq = 0
        self.contents = {}  # key -> value
        self.keyToCount = {}  # key -> count
        self.countToKey = {}  # count -> [ordered set of keys with most recent added at end]

    def _incr(self, key):
        oldCount = self.keyToCount.get(key, 0)
        self.keyToCount[key] = oldCount + 1
        if oldCount > 0:
            keysWithCount = self.countToKey[oldCount]
            del keysWithCount[key]
            if self.minFreq == oldCount and not keysWithCount: # we just incremented the last element of the lowest frequency
                self.minFreq += 1
        if oldCount + 1 in self.countToKey:
            keysWithCount = self.countToKey[oldCount + 1]
            keysWithCount[key] = None # add key with new count to existing ordered hashmap of those counts
        else:
            self.countToKey[oldCount + 1] = OrderedDict({key: None})  # values are unused

    # O(1) time
    def get(self, key: int) -> int:
        if key in self.contents:
            self._incr(key)
            return self.contents[key]
        else:
            return -1

    # O(1) time
    def put(self, key: int, value: int) -> None:
        if key in self.contents:
            # replace
            self._incr(key)
            self.contents[key] = value
            return
        if len(self.contents) == self.capacity:
            # evict
            eSet = self.countToKey[self.minFreq]
            eKey, _ = eSet.popitem(last=False)  # remove from front
            del self.keyToCount[eKey]
            del self.contents[eKey]
        # add
        self._incr(key)
        self.contents[key] = value
        self.minFreq = 1

import unittest

class TestLFUCache(unittest.TestCase):
    def test_lfu_cache(self):
        lfu = LFUCache(2)
        lfu.put(1, 1)  # cache=[1,_], cnt(1)=1
        lfu.put(2, 2)  # cache=[2,1], cnt(2)=1, cnt(1)=1
        self.assertEqual(lfu.get(1), 1)  # return 1, cache=[1,2], cnt(1)=2, cnt(2)=1

        lfu.put(3, 3)  # evict 2 (LFU), cache=[3,1], cnt(3)=1, cnt(1)=2
        self.assertEqual(lfu.get(2), -1)  # return -1 (not found)
        self.assertEqual(lfu.get(3), 3)  # return 3, cache=[3,1], cnt(3)=2, cnt(1)=2

        lfu.put(4, 4)  # evict 1 (LRU among LFU), cache=[4,3], cnt(4)=1, cnt(3)=2
        self.assertEqual(lfu.get(1), -1)  # return -1 (not found)
        self.assertEqual(lfu.get(3), 3)  # return 3, cache=[3,4], cnt(3)=3, cnt(4)=1
        self.assertEqual(lfu.get(4), 4)  # return 4, cache=[4,3], cnt(4)=2, cnt(3)=3


if __name__ == "__main__":
    unittest.main()
