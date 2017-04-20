import random


def construct(num_transactions,
              max_basket_size,
              total_items,
              likely_items,
              likely_draw):
    """Generate a list of transactions.

    Used to analyze apriori and FP-growth algorithms. Implemented as
    a list of sets of non-negative integers.

    Args:
        num_transctions: Number of transactions to generate. Each
            transaction has length chosen from uniform distribution
            of [1, max_basket_size] (inclusive)
        max_basket_size: Maximum size of each transaction.
        total_items: Total number of distinct items appearing in the
            dataset.
        likely_items: The number of items which are most likely
            to appear in the transaction. Should be an integer
            between 1 and total_items
        likely_draw: The chance that one of the items in a transaction
            is a 'likely item', i.e. an integer between 1 and
            likely_items. Must be a number between 0.0 and 1.0,
            but should be greater than likely_items/total_items to
            accurately reflect the intent.

        Returns:
            A list of length num_transactions of sets of integers.
            Each set contains integers in the range 1 to total_items.
    """
    assert 1 <= likely_items < total_items
    assert 0 <= likely_draw <= 1.0
    data = []
    for _ in range(num_transactions):
        basket_size = random.randint(1, max_basket_size)
        basket = set()
        while len(basket) < basket_size:
            r = random.random()
            if r < likely_draw:
                basket.add(random.randint(1, likely_items))
            else:
                basket.add(random.randint(likely_items+1, total_items))
        if basket:
            data.append(basket)
    return data
