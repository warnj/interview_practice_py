import unittest
from leetcode_misc import _all_subarrays_over_2, continuousSubarraysHeap, maxAverageRatioBrute
from sort import quicksort, quicksort2

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


if __name__ == "__main__":
    unittest.main()
