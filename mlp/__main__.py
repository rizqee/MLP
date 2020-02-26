from mlp.perceptron import MultilayerPerceptron

if __name__ == "__main__":
    mlp = MultilayerPerceptron([2, 3, 2], 0.1)
    mlp.fit([1, 2], [1, 0])
