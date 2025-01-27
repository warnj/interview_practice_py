import random


class RandomizedSet:
    def __init__(self):
        self.vals = []
        self.indexes = {}  # val -> index in vals

    def insert(self, val: int) -> bool:
        if val in self.indexes:
            return False
        self.indexes[val] = len(self.vals)
        self.vals.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.indexes:
            return False
        end = self.vals.pop()  # remove from end
        replaceIndex = self.indexes[val]
        del self.indexes[val]
        if val != end:
            self.indexes[end] = replaceIndex  # update the index
            self.vals[replaceIndex] = end  # add where the given val is being removed from
        return True

    def getRandom(self) -> int:
        return self.vals[random.randint(0, len(self.vals)-1)]


obj = RandomizedSet()
print(obj.insert(1))
print(obj.remove(2))
print(obj.getRandom())
