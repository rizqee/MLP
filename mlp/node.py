from abc import ABC, abstractmethod

class Node(ABC):
    @abstractmethod
    def value(self):
        pass

class InputNode(Node):
    def __init__(self, val):
        self.val = val

    def value(self):
        return self.val

class BiasNode(Node):
    def value(self):
        return 1

class OutputNode(Node):
    def __init__(self, bias):
        self.input = { BiasNode() : bias }

    def value(self):
        val = 0
        for node, weight in self.input:
            val += weight * node.value()
        return val
