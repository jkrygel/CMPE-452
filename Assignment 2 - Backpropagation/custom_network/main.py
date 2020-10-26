import random
import math
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Neuron:
    def __init__(self, n=64, lrate=0.5, alpha=0):
        self.weight = []
        for i in range(n):
            self.weight.append(random.uniform(-1,1))
        self.learnrate = lrate

        self.alpha = alpha                              #for momentum
        self.prev_delta_weight = [0]*len(self.weight)   #for momentum

        self.y = 0                                      #output
        self.prev_weight = [0]*len(self.weight)
        self.de = 0

    def sigmoid(self, a):
        try:
            ex = math.exp(-a)
        except OverflowError:
            ex = float('inf')
        return 1/(1 + ex)

    def calc_a(self, x):
        a = 0
        for i in range(len(x)):
            a += self.weight[i] * x[i]
        return a

    def calc_output(self, x):
        a = self.calc_a(x)
        self.y = self.sigmoid(a)
        
        return self.y

    #backprop_error = backpropagated values from next layer (d-y) for output layer
    def learn(self, backprop_error, x):
        delta_weight = [0]*len(self.weight)
        f_prime = (self.y * (1 - self.y))
        self.de = f_prime * backprop_error

        for i in range(len(self.weight)):
            delta_weight[i] = self.learnrate * x[i] * f_prime * backprop_error
            self.prev_weight[i] = self.weight[i]
            self.weight[i] +=   delta_weight[i] + \
                                self.alpha * self.prev_delta_weight[i]
            self.prev_delta_weight[i] = delta_weight[i]

    def print_weight(self):
        print("\n")
        for i in range(len(self.weight)):
            print("Weight " + str(i) + ": ", end="")
            print("%.3f" % self.weight[i])
        # print("\n")


class DummyNeuron:
    def __init__(self):
        self.y = 1

    def calc_output(self, x):
        return self.y


class InputNeuron:
    def __init__(self):
        self.y = 0

    def calc_output(self, x):
        return self.y

    def input_data(self, x):
        self.y = x


class NeuralNetwork:
    def __init__(self, n):
        self.layer = []
        for i in range(n+1):
            self.layer.append([])
            # add dummy neuron for x0 to all layers except output layer
            if i < n:
                self.layer[i].append(DummyNeuron())

    def add_layer(self, layer_n, node_n, lrate, alpha):
        for i in range(node_n):
            if layer_n == 0:
                self.layer[layer_n].append(InputNeuron())
            else:
                self.layer[layer_n].append(Neuron(len(self.layer[layer_n-1]), \
                                                lrate, alpha))

    def input_train_data(self, data, d_output):
        # put data into input layer (except x0)
        for i in range(1, len(self.layer[0])):
            self.layer[0][i].input_data(data[i-1])

        # for every layer after input layer
        for i in range(1, len(self.layer)):
            x = self.get_layer_output(i-1)
            #for every node in layer
            for j in range(len(self.layer[i])):
                self.layer[i][j].calc_output(x)

        # check for error and adjust weight for output layer
        i_output = len(self.layer)-1
        sum_correct = 0
        for i in range(len(self.layer[i_output])):
            d = 0
            if i == d_output:
                d = 1
            else:
                d = 0
            de = d - self.layer[i_output][i].y
            if de == 0:
                sum_correct += 1
                continue
            else:
                x = self.get_layer_output(i_output-1)
                self.layer[i_output][i].learn(de, x)

        # print("Starting to train hidden layer..")
        for i in reversed(range(1, len(self.layer)-1)):
            for j in range(1, len(self.layer[i])):
                # print("Training hidden layer " + str(i) + "\tnode no. " + str(j))
                de = self.get_layer_error(i+1, j)
                x = self.get_layer_output(i-1)
                self.layer[i][j].learn(de, x)
        return sum_correct

    def get_layer_output(self, layer_n):
        y = []
        for i in range(len(self.layer[layer_n])):
            y.append(self.layer[layer_n][i].y)
        return y

    # get de of all layers above it
    def get_layer_error(self, layer_n, node_n):
        sum = 0
        for i in range(1, len(self.layer[layer_n])):
            sum +=  self.layer[layer_n][i].de * \
                    self.layer[layer_n][i].prev_weight[node_n]
        return sum

    def test(self, data):
        y = []
        # put data into input layer (except x0)
        for i in range(1, len(self.layer[0])):
            self.layer[0][i].input_data(data[i-1])

        # for every layer after input layer
        for i in range(1, len(self.layer)):
            x = self.get_layer_output(i-1)
            #for every node in layer
            for j in range(len(self.layer[i])):
                if (i == len(self.layer)-1 ):
                    y.append(self.layer[i][j].calc_output(x))
                else:
                    self.layer[i][j].calc_output(x)
        return y

    def export_weights(self, filename):
        output_text = ""
        for i in range(1, len(self.layer)):
            for j in range(1, len(self.layer[i])):
                output_text += "# Layer: " + str(i) + "\tNode:" + str(j) + "\n"
                output_text += str(self.layer[i][j].weight) + "\n\n"
        f = open(filename, 'w')
        f.write(output_text)
        f.close()


if __name__ == '__main__':
    layer_config = [784, 10, 10]
    learning_rate = 0.001
    alpha = 0.1
    train_iterations = 5

    # Load MNIST training data with pytorch methods
    transform=transforms.Compose([
        transforms.ToTensor(),
        ])
    train_dataset = datasets.MNIST('./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

    dat = next(iter(train_loader))[0].numpy().reshape((60000,784))
    output = next(iter(train_loader))[1].numpy()

    nn = NeuralNetwork(len(layer_config)-1)
    for i in range(len(layer_config)):
        nn.add_layer(i, layer_config[i], learning_rate, alpha)

    print("Network config = [", end="")
    for i in range(len(layer_config)):
        print(str(layer_config[i]) + ",", end="")
    print("\b]")
    print("Start training..")
    start_time = time.time()

    # Train NN using training dataset
    for n in range(train_iterations):
        sum_correct = 0
        # for i in range(len(dat)):
        for i in range(10000):
            sum_correct += nn.input_train_data(dat[i], output[i])
            percent_correct = sum_correct/(i+1)
            if i % 1000 == 0 and i != 0:
                print("Hit 1000")
        print (f"Epoch: {n}")

    elapsed_time = time.time() - start_time
    #nn.export_weights("./weights.dat")
    print("Training completed!", end="\t")
    print("Total training time: " + "%.2f"%(elapsed_time) +"s \t")
    print("Start testing.")
    y = []
    classified_correct = [0]*10
    total_num = [0]*10
    sum_correct = 0
    
    # Load MNIST testing data with pytorch methods
    transform=transforms.Compose([
        transforms.ToTensor(),
        ])
    train_dataset = datasets.MNIST('./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

    dat = next(iter(train_loader))[0].numpy().reshape((60000,784))
    output = next(iter(train_loader))[1].numpy()

    # Test NN using testing dataset
    for i in range(len(dat)):
            y = nn.test(dat[i])
            maxindex = 0
            for j in range(len(y)):
                if y[j] > y[maxindex]:
                    maxindex = j
            desired_output = output[i]
            total_num[desired_output] += 1
            if (y[desired_output] == 1) or (desired_output == maxindex):
                classified_correct[desired_output] += 1
                sum_correct += 1
                # print("Correct output: " + str(sum_correct))

    pertg_correct = sum_correct/len(dat) * 100
    print("Testing completed.")
    # print("Accuracy: " + str(pertg_correct) + "%")

    output_text = ""
    output_text += ('='*30 + " Configuration " + '='*30 + "\n")
    output_text += ("BPN layers:  \t" + str(layer_config) + "\n")
    output_text += ("Learning rate: \t" + str(learning_rate) + "\n")
    output_text += ("Momentum(α): \t" + str(alpha) + "\n")
    output_text += ("No. Iteration:\t" + str(train_iterations) + "\n\n")

    output_text += ('='*30 + " Result " + '='*30 + "\n")
    output_text += ("No. of correctly classified digits:\n")
    for i in range(len(classified_correct)):
        output_text += "Digit " + str(i) + ": " + str(classified_correct[i]) + \
                        "/" + str(total_num[i]) + "   \t("\
                        "%.2f"%(classified_correct[i]/total_num[i]*100) + "%)\n"
    output_text += ("Average accuracy:\t" + "%.3f"%(pertg_correct) + "%\n")
    output_text += ("\n")

    print(output_text)