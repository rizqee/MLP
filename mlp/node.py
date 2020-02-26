from mlp.common import sigmoid

from abc import ABC, abstractmethod


class Node(ABC):
    @abstractmethod
    def value(self):
        pass


class InputNode(Node):
    def __init__(self, val=0):
        self.val = val

    def value(self):
        return self.val

    def set_value(self, val):
        self.val = val


class BiasNode(Node):
    def __init__(self):
        self.error = 0

    def value(self):
        return 1


class OutputNode(Node):
    def __init__(self, bias=0):
        self.input = {BiasNode(): [bias, 0]}
        self.val = bias
        self.error = 0

    def add_child(self, node, weight=0):
        self.input[node] = [weight, 0]

    def add_weight(self, node, weight):
        self.input[node][1] += weight

    def update_weight(self):
        for node in self.input:
            self.input[node][0] += self.input[node][1]
            self.input[node][1] = 0

    def value(self):
        return self.val

    def calc_value(self):
        val = 0
        for node in self.input:
            val += self.input[node][0] * node.value()
        self.val = sigmoid(val)
