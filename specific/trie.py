class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

# https://leetcode.com/problems/design-add-and-search-words-data-structure
class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word):
        cur = self.root
        for character in word:
            cur = cur.children.setdefault(character, TrieNode())
        cur.is_word = True

    def search(self, word):
        def dfs(node, index):
            if index == len(word):
                return node.is_word
            if word[index] == ".":
                for child in node.children.values():
                    if dfs(child, index + 1):
                        return True
            if word[index] in node.children:
                return dfs(node.children[word[index]], index + 1)
            return False
        return dfs(self.root, 0)

class WordDictionarySoln:
    def __init__(self):
        self.trie = {}

    def addWord(self, word: str) -> None:
        node = self.trie

        for ch in word:
            if not ch in node:
                node[ch] = {}
            node = node[ch]
        node["$"] = True

    def search(self, word: str) -> bool:
        def search_in_node(word, node) -> bool:
            for i, ch in enumerate(word):
                if not ch in node:
                    # if the current character is '.'
                    # check all possible nodes at this level
                    if ch == ".":
                        for x in node:
                            if x != "$" and search_in_node(
                                word[i + 1 :], node[x]
                            ):
                                return True
                    # If no nodes lead to an answer
                    # or the current character != '.'
                    return False
                # If the character is found
                # go down to the next level in trie
                else:
                    node = node[ch]
            return "$" in node
        return search_in_node(word, self.trie)

# example with words:  bay, bad, mad
#       b    m
#      a      a
#    d  y       d
class TrieNodeUgly:
    def __init__(self, c=''):
        self.c = c  # current character (empty for root)
        self.end = False  # end of a word
        self.next = [None] * 26  # next letters

class WordDictionaryUgly:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        # look for leaf node
        cur = self.root
        for j in range(len(word)):
            c = word[j]
            i = ord(c) - ord('a')
            ptr = cur.next[i]
            if not ptr:
                # found the end, create it
                new = TrieNode(c)
                if j == len(word) - 1:
                    new.end = True
                cur.next[i] = new
                cur = new
            else:  # progress down the tree
                cur = ptr
                if j == len(word) - 1:
                    ptr.end = True

    def _search(self, word, cur):
        if not word:
            return True
        if not cur:
            return False

        for j in range(len(word)):
            c = word[j]
            if c == '.':
                if j == len(word) - 1:
                    for ptr in cur.next:  # if any children exist that end a word, the result is true
                        if ptr and ptr.end:
                            return True
                else:
                    for ptr in cur.next:
                        if self._search(word[j + 1:], ptr):
                            return True
                return False
            else:
                ptr = cur.next[ord(c) - ord('a')]
                if not ptr:
                    return False
                else:  # progress down the tree
                    if j == len(word) - 1 and ptr.end:
                        return True
                    cur = ptr
        return False

    def search(self, word: str) -> bool:
        return self._search(word, self.root)


import unittest

class TestWordDictionary(unittest.TestCase):
    def test_word_dictionary(self):
        wordDictionary = WordDictionary()
        wordDictionary.addWord("bad")
        wordDictionary.addWord("dad")
        wordDictionary.addWord("mad")

        self.assertFalse(wordDictionary.search("pad"))  # Not added
        self.assertTrue(wordDictionary.search("bad"))  # Exact match
        self.assertTrue(wordDictionary.search(".ad"))  # Wildcard match
        self.assertTrue(wordDictionary.search("b.."))  # Wildcard match
        self.assertTrue(wordDictionary.search("..."))  # Wildcard match

    def test_word_dictionary(self):
        wordDictionary = WordDictionary()
        wordDictionary.addWord("a")
        wordDictionary.addWord("a")
        self.assertTrue(wordDictionary.search("."))  # Wildcard match
        self.assertTrue(wordDictionary.search("a"))  # Exact match
        self.assertFalse(wordDictionary.search("aa"))  # No match
        self.assertTrue(wordDictionary.search("a"))  # Exact match
        self.assertFalse(wordDictionary.search(".a"))  # No match
        self.assertFalse(wordDictionary.search("a."))  # No match

if __name__ == "__main__":
    unittest.main()
