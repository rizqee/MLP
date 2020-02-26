from mlp.node import Node, InputNode, BiasNode, OutputNode

from math import exp


class MultilayerPerceptron:
    def __init__(self, layer):
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

    def backward_prop(self, target):
        if len(target) != len(self.nodes[-1]):
            raise ValueError("target must have the same length as the last layer")

    def update_weight(self):
        for lvl, layer in enumerate(self.nodes):
            if lvl == 0:
                continue
            else:
                for node in layer:
                    node.update_weight()
