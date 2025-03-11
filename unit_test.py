import unittest

from array import ArrayPractice
from leetcode_misc import _all_subarrays_over_2, continuousSubarraysHeap, maxAverageRatioBrute, mySqrt
from sort import quicksort, quicksort2
from array2d import getLargestRhombusSum

class TestLeetcodeMisc(unittest.TestCase):
    def test_all_subarrays_over_2(self):
        self.assertEqual(_all_subarrays_over_2([1, 2, 3]), [[1, 2],  [1, 2, 3], [2, 3]])
        self.assertEqual(_all_subarrays_over_2([1, 2]), [[1, 2]])
        self.assertEqual(_all_subarrays_over_2([1]), [])
        self.assertEqual(_all_subarrays_over_2([]), [])

    def test_continuous_subarrays(self):
        self.assertEqual(continuousSubarraysHeap(None, [1,2,3]), 6)
        self.assertEqual(continuousSubarraysHeap(None, [5,4,2,4]), 8)

    def test_max_avg_ratio(self):
        self.assertEqual(maxAverageRatioBrute(None, [1,2,3]), 6)

    def test_max_average_ratio(self):
        self.assertAlmostEqual(maxAverageRatioBrute(None, [[1, 2], [3, 5], [2, 2]], 2), 0.78333, places=5)

class TestSort(unittest.TestCase):
    def test_quicksort(self):
        for i in range(10):
            a = [3,2,1]
            quicksort(a)
            self.assertEqual([1,2,3], a)
            a = [4,9,1,2]
            quicksort(a)
            self.assertEqual([1,2,4,9], a)
        for i in range(10):
            a = [3,2,1]
            quicksort2(a)
            self.assertEqual([1,2,3], a)
            a = [4,9,1,2]
            quicksort2(a)
            self.assertEqual([1,2,4,9], a)

class TestMySqrt(unittest.TestCase):
    def test_mySqrt(self):
        test_cases = [
            (0, 0),
            (1, 1),
            (4, 2),
            (8, 2),
            (9, 3),
            (16, 4),
            (25, 5),
            (26, 5),
            (99, 9),
            (100, 10),
            (101, 10),
            (2147395599, 46339),  # Large test case
        ]
        for x, expected in test_cases:
            with self.subTest(x=x):
                self.assertEqual(mySqrt(x), expected)

class ArrayTesting(unittest.TestCase):
    a = ArrayPractice()

    def testMaxSubarraySumCircular(self):
        self.assertEqual(3, self.a.maxSubarraySumCircular([1, -2, 3, -2]))
        self.assertEqual(10, self.a.maxSubarraySumCircular([5, -3, 5]))
        self.assertEqual(-2, self.a.maxSubarraySumCircular([-3, -2, -3]))

    def testSwapOnceIncr(self):
        self.assertTrue(self.a.swapOnceIncr([1, 5, 10, 20]))
        self.assertTrue(self.a.swapOnceIncr([1, 3, 900, 10]))
        self.assertFalse(self.a.swapOnceIncr([13, 31, 30]))
        self.assertFalse(self.a.swapOnceIncr([20, 10, 9, 8]))
        self.assertTrue(self.a.swapOnceIncr([91, 20]))
        self.assertTrue(self.a.swapOnceIncr([17, 18, 91, 20]))

    def testConcatSum(self):
        self.assertEqual(198, self.a.concatSum([1, 2, 3]))
        self.assertEqual(1344, self.a.concatSum([10, 2]))

    def testLongestCommonPrefix(self):
        self.assertEqual('fl', self.a.longestCommonPrefix(["flower", "flow", "flight"]))

    def testLongestCommonPrefixTwoLists(self):
        self.assertEqual(3, self.a.longestCommonPrefixTwoLists(["flower", "flow", "finger"],['advice', 'tweet', 'finance']))
        self.assertEqual(2, self.a.longestCommonPrefixTwoLists(["flower", "flow", "finger"],['advice', 'tweet', 'flip']))


class TestArray2D(unittest.TestCase):
    def test_getLargestRhombusSum(self):
        self.assertEqual(27, getLargestRhombusSum([[1, 2, 3, 4], [6, 7, 8, 9], [4, 5, 6, 7], [9, 8, 7, 6]], 2))
        # example: size=3
        # [1,2,3,4,5,7]
        # [6,7,8,9,2,4]  -> max(3+7+4+8+7+6+8+9, 4+8+5+7+6+7+9+2) -> max(52,48) -> 52
        # [4,5,6,7,8,9]
        # [9,8,7,6,7,1]
        # [9,8,7,6,7,1]
        self.assertEqual(52, getLargestRhombusSum(
            [[1, 2, 3, 4, 5, 7], [6, 7, 8, 9, 2, 4], [4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 7, 1], [9, 8, 7, 6, 7, 1]], 3))

if __name__ == "__main__":
    unittest.main()
