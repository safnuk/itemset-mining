import collections

import dataset
import trie


class FpGrowth:
    def __init__(self, T, supp, multiplicities=None):
        """Use the fp growth algorithm to find all frequent itemsets.

        Based off of algorithm description from
        http://hareenlaks.blogspot.com/2011/10/how-to-identify-frequent-patterns-from.html

        and

        http://hareenlaks.blogspot.com/2011/06/fp-tree-example-how-to-identify.html

        Args:
            T: A list of transactions, where each transaction is a set of
                positive integers
            supp: Measure the minimum support count needed for an itemset to
                be considered frequent.
            multiplicities: Either None or a list of integers the same length
                as T, indicating how many times the corresponding
                transaction is reqpeated. If None, each transaction
                is assumed to have multiplicity 1.
        """
        self.T = T
        self.supp = supp
        self.multiplicities = multiplicities if multiplicities else [1] * len(T)
        self.frequent_itemsets = []
        self._count_items()
        self._build_trie()
        self._find_frequent_itemsets()

    def _count_items(self):
        counter = collections.Counter()
        for t, m in zip(self.T, self.multiplicities):
            for item in t:
                counter[item] += m
        self.sort_order = {k: v for (k, v) in counter.most_common()}
        self.frequent_items_sorted = [k for (k, v) in counter.most_common()
                                      if v >= self.supp]
        # use a set for faster item lookups
        self.frequent_items = set(self.frequent_items_sorted)

    def _build_trie(self):
        self.root = trie.Node()
        for t, m in zip(self.T, self.multiplicities):
            items = sorted(t, key=lambda item: self.sort_order[item],
                           reverse=True)
            items = [item for item in items if item in self.frequent_items]
            self.root.insert(items, m)

    def _find_frequent_itemsets(self):
        conditional_patterns = collections.defaultdict(list)
        for node in self.root.node_list():
            conditional_patterns[node.item].append(node)
        # work from least frequent items to most frequent
        for item in reversed(self.frequent_items_sorted):
            itemsets = self._find_itemsets_containing(
                conditional_patterns[item])
            for freq in itemsets:
                self.frequent_itemsets.append(freq | {item})

    def _find_itemsets_containing(self, patterns):
        transactions = []
        multiplicities = []
        for node in patterns:
            transaction = node.expand_to_root()
            if transaction:
                transactions.append(transaction)
                multiplicities.append(node.count)
        f = FpGrowth(transactions, self.supp, multiplicities)
        return [set()] + f.frequent_itemsets


if __name__ == '__main__':
    T = dataset.construct(10, 4, 6, 2, 0.7)
    F = FpGrowth(T, 3)
    print(T)
    print(F.frequent_itemsets)
