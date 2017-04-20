class Node:
    def __init__(self, parent=None, item=None, multiplicity=1):
        self.parent = parent
        self.item = item
        if item is not None:
            self.count = multiplicity
        self.children = []

    def insert(self, items, multiplicity=1):
        """Insert the list of items into the trie.

        Assumes that the list is in sorted order (most common first)

        Args:
            items: list of items to be inserted
        """
        if not items:
            return
        item = items[0]
        for child in self.children:
            if child.item == item:
                child.count += multiplicity
                child.insert(items[1:], multiplicity)
                return
        # item not a child, so add a new child
        new_child = Node(self, item, multiplicity)
        self.children.append(new_child)
        new_child.insert(items[1:], multiplicity)

    def node_list(self):
        nodes = []
        queue = list(self.children)
        while queue:
            node = queue.pop()
            nodes.append(node)
            queue = queue + node.children
        return nodes

    def expand_to_root(self):
        if self.parent.item is None:
            return []
        else:
            return [self.parent.item] + self.parent.expand_to_root()

    def __str__(self):
        output = "["
        for c in self.children:
            output = output + '({}, {})'.format(c.item, c.count)
        return output + ']'

    def __repr__(self):
        return self.__str__()
