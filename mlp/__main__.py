from mlp.perceptron import MultilayerPerceptron
from mlp.node import BiasNode

if __name__ == "__main__":
    bias_node_1 = BiasNode()
    bias_node_2 = BiasNode()
    if id(bias_node_1) == id(bias_node_2):
        print('same id')
    mlp = MultilayerPerceptron([2, 2, 2], 0.5)
    mlp.nodes[1][0].input[mlp.nodes[0][0]][0] = 0.15
    mlp.nodes[1][0].input[mlp.nodes[0][1]][0] = 0.2
    mlp.nodes[1][1].input[mlp.nodes[0][0]][0] = 0.25
    mlp.nodes[1][1].input[mlp.nodes[0][1]][0] = 0.3
    mlp.nodes[2][0].input[mlp.nodes[1][0]][0] = 0.4
    mlp.nodes[2][0].input[mlp.nodes[1][1]][0] = 0.45
    mlp.nodes[2][1].input[mlp.nodes[1][0]][0] = 0.5
    mlp.nodes[2][1].input[mlp.nodes[1][1]][0] = 0.55

    mlp.nodes[1][0].input[BiasNode()][0] = 0.35
    mlp.nodes[1][1].input[BiasNode()][0] = 0.35
    mlp.nodes[2][0].input[BiasNode()][0] = 0.6
    mlp.nodes[2][1].input[BiasNode()][0] = 0.6
    
    mlp.fit([0.05, 0.1], [0.01, 0.99])
    
    for lvl, layer in enumerate(mlp.nodes):
        print('Layer ' + str(lvl))
        for i, node in enumerate(layer):
            print('Node ' + str(i) + ' Value : ', end='')
            print(node.value())
    mlp.update_weight()
    mlp.print_weight()
