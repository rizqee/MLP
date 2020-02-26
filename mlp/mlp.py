from mlp.node import Node, InputNode, BiasNode, OutputNode

from math import exp

class MultilayerPerceptron:
    def __init__(self, layer):
        if not layer:
            raise ValueError('layer must be a nonempty array')

        prev_layer = []
        for i, cnt in enumerate(layer):
            if i == 0:
                layer = [InputNode(0) for j in range(layer[i])]
                self.input = layer
            else:
                layer = [OutputNode(0) for j in range(layer[i])]
            
            if i == len(layer) - 1:
                self.output = layer
            
            prev_layer = layer

    def forward_prop(self):


    @staticmethod
    def f(x):
        return 1 / (1 + exp(-x))