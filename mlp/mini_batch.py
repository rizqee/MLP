from mlp.perceptron import MultilayerPerceptron
from itertools import chain 


class MiniBatch:
    def __init__(self, data, error_threshold, mini_epoch_num, learning_rate, hidden_layer, max_iteration):
        self.target = []
        for row in data:
            self.target.append((row.pop(len(row)-1)))
        self.input = data
        input_layer = len(self.input[0])
        self.target_layer = 1
        if self.target_contains_not_number():
            self.change_target() 
        self.error_threshold = error_threshold
        self.mini_epoch_num = mini_epoch_num
        self.max_iteration = max_iteration
        self.mlp = MultilayerPerceptron([input_layer] + hidden_layer + [self.target_layer], learning_rate)
        
        self.fit()
    
    def fit(self):
        i = 0
        iteration_num = 0
        while (iteration_num == 0 or self.mlp.error() > self.error_threshold) and iteration_num <= self.max_iteration: 
            for j in range(min(self.mini_epoch_num, len(self.input) - i)):
                self.mlp.fit(self.input[i], self.target[i])
                i += 1
            if i >= len(self.input):
                i = 0
            self.mlp.update_weight()
            iteration_num += 1
    
    def print_weight(self):
        self.mlp.print_weight()

    def change_target(self): 
        self.unique_target = list(set(self.target))
        self.target_layer = len(self.unique_target)
        for i in range(len(self.target)):
            temp_element = self.target[i]
            self.target[i] = [0] * self.target_layer
            self.target[i][self.unique_target.index(temp_element)] = 1
        
    def target_contains_not_number(self):
        exist = False
        for element in self.target:
            if not isinstance(element, (int, float)):
                exist = True
        return exist
    
    def classify(self, input):
        output = self.mlp.produce_output(input)
        return self.unique_target[output.index(max(output))]
