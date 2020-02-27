from mlp.node import Node, InputNode, BiasNode, OutputNode
import numpy as np

class MultilayerPerceptron:
    def __init__(self, layer, learning_rate):
        if not layer:
            raise ValueError("layer must be a nonempty array")
        
        self.input_layer = layer[0]
        self.output_layer = layer[-1]

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

        self.error_sum = 0

        self.randomize_weight()

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
                    node.error = val - node.value()

            if lvl != 0:
                for node in layer:
                    node.error *= node.value() * (1 - node.value())
                    for child in node.input:
                        node.add_weight(
                            child, self.learning_rate * node.error * child.value()
                        )

            if lvl >= 2:
                for node in layer:
                    for child in node.input:
                        if isinstance(child, OutputNode):
                            child.error += node.input[child][0] * node.error

    def produce_output(self, input):
        self.forward_prop(input)
        return [node.value() for node in self.nodes[-1]]

    def fit(self, input, target):
        self.forward_prop(input)
        self.backward_prop(target)
        self.update_error(target)

    def error(self):
        return self.error_sum

    def update_error(self, target):
        for i, node in enumerate(self.nodes[-1]): 
            self.error_sum += (target[i] - node.value())**2 / 2

    def reset_error(self):
        self.error_sum = 0

    def update_weight(self):
        for lvl, layer in enumerate(self.nodes):
            if lvl == 0:
                continue
            else:
                for node in layer:
                    node.update_weight()

    def print_weight(self):
        for lvl, layer in enumerate(self.nodes):
            if lvl == 0:
                continue
            if lvl == len(self.nodes) - 1:
                print('Layer Output')    
            else:
                print('Layer Hidden ' + str(lvl))
            for i, node in enumerate(layer):
                for j, prev_node in enumerate(self.nodes[lvl-1]):
                    if lvl == 1:
                        print('  Weight In' + str(j), end=' ')
                    else:
                        print('  Weight H' + str(lvl-1) + ',' + str(j), end='')
                    if lvl == len(self.nodes) - 1 :
                        print(' - Out' + str(i) + ': ', end='')
                    else:
                        print(' - H' + str(lvl) + ',' + str(i) + ': ', end='')
                    print(node.input[prev_node][0])
                print('  Weight Bias', end='')
                if lvl == len(self.nodes) - 1 :
                    print(' - Out' + str(i) + ': ', end='')
                else:
                    print(' - H' + str(lvl) + ',' + str(i) + ': ', end='')
                print(node.input[BiasNode()][0])

    def randomize_weight(self):
        for lvl, layer in enumerate(self.nodes):
            if lvl == 0:
                continue
            for i, node in enumerate(layer):
                for j, prev_node in enumerate(self.nodes[lvl-1]):
                    node.input[prev_node][0] = np.random.normal()*np.sqrt(1/(self.input_layer + self.output_layer))
                node.input[BiasNode()][0] = np.random.normal()*np.sqrt(1/(self.input_layer + self.output_layer))                                    
                    
