from mlp.perceptron import MultilayerPerceptron

if __name__ == "__main__":
    mlp = MultilayerPerceptron([2, 3, 2])
    mlp.forward_prop([2, 3])
