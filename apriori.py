import collections

import dataset


class Apriori:
    def __init__(self, T, supp):
        """Use the apriori algorithm to find all frequent itemsets.

        Args:
            T: A list of transactions, where each transaction is a set of
                positive integers
            supp: Measure the minimum support count needed for an itemset to
                be considered frequent.

        Returns:
            A list of sets of frequent itemsets. The k-th element in the
                list is the set of k-frequent itemsets.
        """
        self.T = T
        self.supp = supp
        self.L = [set()]  # L[0] is the empty set
        self._find_frequent_items()
        while self.L[-1]:
            itemset_count = collections.Counter()
            C_k = self._candidate_set(len(self.L))
            for basket in T:
                candidates = {c for c in C_k if c <= basket}
                itemset_count.update(candidates)
            self.L.append({c for c in C_k if itemset_count[c] >= supp})
        self.L.pop()  # remove the last (empty) set

    def _find_frequent_items(self):
        """Find all frequent 1-itemsets (i.e. individual items)."""
        count = collections.Counter()
        for t in self.T:
            count.update(t)
        self.L.append({frozenset({item})
                       for item in count
                       if count[item] >= self.supp})

    def _candidate_set(self, k):
        """Use apriori principle to generate all candidate frequent k-itemsets.

        In particular, it restricts to k-itemsets with all subsets
        being frequent.
        """
        assert 2 <= k
        assert k == len(self.L)
        C_k = set()
        for a in self.L[k-1]:
            for b in self.L[1]:
                if b.isdisjoint(a):
                    C_k.add(a | b)
        C_k_iterator = set(C_k)  # make a copy so we can modify the original
        for a in C_k_iterator:
            for e in a:
                if (a - {e}) not in self.L[k-1]:
                    C_k.remove(a)
                    break
        return C_k

if __name__ == '__main__':
    T = dataset.construct(100, 4, 6, 2, 0.7)
    A = Apriori(T, 3)
    print(T)
    print(A.L)
