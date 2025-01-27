class LRUCache:
    class Node:
        def __init__(self, key=0, val=0, n=None, p=None):
            self.key = key
            self.val = val
            self.next = n
            self.prev = p
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.nodes = {}  # all values without ordering. key -> node with val
        self.head = None  # doubly linked list from most to least recently used
        self.tail = None  # can use dummy head and tail nodes to avoid some of the edge cases below

    # moves node to front
    def _moveFront(self, node):
        if self.head != node:
            node.prev.next = node.next
            if node.next:
                node.next.prev = node.prev
            else:
                self.tail = node.prev
            node.next = self.head
            self.head.prev = node
            node.prev = None
            self.head = node

    # O(1)
    def get(self, key: int) -> int:
        if key in self.nodes:
            node = self.nodes[key]
            self._moveFront(node)
            return node.val
        return -1

    # O(1)
    def put(self, key: int, value: int) -> None:
        if key in self.nodes:
            # move to front
            self.nodes[key].val = value
            self._moveFront(self.nodes[key])
        else:
            if len(self.nodes) == self.capacity:
                # evict tail
                del self.nodes[self.tail.key]
                if self.tail == self.head:  # special case max size is 1 we need to re-init tail below to be newNode
                    self.head = None
                self.tail = self.tail.prev
                if self.tail:
                    self.tail.next = None
            # create new at front
            newNode = self.Node(key, value, self.head, None)
            if self.head:
                self.head.prev = newNode
            else:
                self.tail = newNode  # first node both head and tail
            self.head = newNode
            self.nodes[key] = newNode

obj = LRUCache(1)
print(obj.get(2))
obj.put(2,2)
print(obj.get(2))
obj.put(3,3)
obj.put(4,4)
print(obj.get(2))
