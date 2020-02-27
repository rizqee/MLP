from mlp.node import Node, InputNode, BiasNode, OutputNode

from math import exp


class MultilayerPerceptron:
    def __init__(self, layer, learning_rate):
        if not layer:
            raise ValueError("layer must be a nonempty array")

        self.nodes = []
        prev_layer = []
        for lvl, cnt in enumerate(layer):
            layer = []
            if lvl == 0:
                layer = [InputNode() for i in range(cnt)]
            else:
                for i in range(cnt):
                    node = OutputNode()
                    for child in prev_layer:
                        node.add_child(child)
                    layer.append(node)

            prev_layer = layer
            self.nodes.append(prev_layer)

        self.learning_rate = learning_rate

    def forward_prop(self, input):
        if len(input) != len(self.nodes[0]):
            raise ValueError("input must have the same length as the first layer")

        for lvl, layer in enumerate(self.nodes):
            if lvl == 0:
                for node, val in zip(layer, input):
                    node.set_value(val)
            else:
                for node in layer:
                    node.calc_value()
                    node.error = 0

    def backward_prop(self, target):
        if len(target) != len(self.nodes[-1]):
            raise ValueError("target must have the same length as the last layer")

        for lvl, layer in reversed(list(enumerate(self.nodes))):
            if lvl == len(self.nodes) - 1:
                for node, val in zip(layer, target):
                    node.error = (
                        node.value() * (1 - node.value()) * (val - node.value())
                    )

            if lvl >= 2:
                for node in layer:
                    for child in node.input:
                        if isinstance(child, OutputNode):
                            child.error += node.input[child][0] * node.error

            if lvl != 0:
                for node in layer:
                    node.error *= node.value() * (1 - node.value())
                    for child in node.input:
                        node.add_weight(
                            child, self.learning_rate * node.error * child.value()
                        )

    def fit(self, input, target):
        self.forward_prop(input)
        self.backward_prop(target)

    def update_weight(self):
        for lvl, layer in enumerate(self.nodes):
            if lvl == 0:
                continue
            else:
                for node in layer:
                    node.update_weight()
