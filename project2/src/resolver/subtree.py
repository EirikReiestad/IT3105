class SubTree:
    def __init__(self, root):
        self.root = root
        self.nodes: list(SubTree) = []

    def add_node(self, node):
        self.nodes.append(node)

    def get_node(self, name):
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_root(self):
        return self.root

    def get_nodes(self):
        return self.nodes

    def __str__(self):
        return self.root

    def __repr__(self):
        return self.root
