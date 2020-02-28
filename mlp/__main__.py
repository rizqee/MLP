from mlp.perceptron import MultilayerPerceptron
from mlp.node import BiasNode
from mlp.mini_batch import MiniBatch

import copy
import csv
from random import shuffle

if __name__ == "__main__":
    data = []
    with open("iris.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        data.pop(0)
        shuffle(data)

    for i in range(len(data)):
        for j in range(len(data[0]) - 1):
            data[i][j] = float(data[i][j])

    training_data = copy.deepcopy(data)
    len_training = int(0.8 * len(data))
    training_data = training_data[:len_training]
    testing_data = data[len_training:]

    mb = MiniBatch(copy.deepcopy(training_data), 0.001, 20, 0.1, [4, 2, 1], 25000)
    # mb.print_weight()

    test_count = 0
    for row in testing_data:
        print(row)
        cls = mb.classify(row[:-1])
        if cls == row[-1]:
            test_count += 1
        print(cls)

    train_count = 0
    for row in training_data:
        print(row)
        cls = mb.classify(row[:-1])
        if cls == row[-1]:
            train_count += 1
        print(cls)

    print("Accuracy Training : ", float(train_count) / len(training_data) * 100, " %")
    print("Accuracy Testing : ", float(test_count) / len(testing_data) * 100, " %")
