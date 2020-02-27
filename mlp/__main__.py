from mlp.perceptron import MultilayerPerceptron
from mlp.node import BiasNode
from mlp.mini_batch import MiniBatch
import copy
import csv

if __name__ == "__main__":
    data=[]
    with open('iris.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        data.pop(0)
    
    for i in range(len(data)):
        for j in range(len(data[0]) - 1):
            data[i][j] = float(data[i][j]) 
    
    original_data = copy.deepcopy(data)

    mb = MiniBatch(data, 0.001, 5, 0.1, [2,2,2], 99999)
    #mb.print_weight()
    

    count = 0
    for row in original_data:
        print(row)
        cls = mb.classify(row[:-1])
        if cls == row[-1]:
            count += 1
        print(cls)
    print('Accuracy : ', float(count)/len(original_data) * 100, ' %')    
    