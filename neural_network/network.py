# Thic program implements a feed forward NN from scratch in python.

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--gamma', type=float, required=True)
parser.add_argument('--d', type=float, required=True)
parser.add_argument('--hidden_dim', type=int, required=True)
parser.add_argument('--init', choices=['random', 'zero'], required=True)


args = parser.parse_args()


class Layer:
    def __init__(self, input_dim, output_dim, init):
        '''
        Each layer has a set of {output_dim} neurons each with a set of {input_dim + 1}  weights (1 for each input plus 1 bias weight). 
        '''
        if init == 'random':
            self.neurons = [{"weights": np.random.rand(input_dim + 1)} for i in range(output_dim)]
        elif init == 'zero':
            self.neurons = [{"weights": np.zeros(input_dim + 1)} for i in range(output_dim)]


    def __str__(self):
        return str(self.neurons)

    def __len__(self):
        return len(self.neurons)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i == self.__len__():
            raise StopIteration
        else:
            item = self.neurons[self.i]
            self.i += 1
            return item 

    def __getitem__(self,index):
        return self.neurons[index]

    def sum_and_activate(self, inputs):
        '''
        compute and return the output value for a neuron.
        '''
        next_inputs = []
        for neuron in self.neurons:
            bias = neuron['weights'][-1]
            weighted_sum = bias + np.sum(neuron['weights'][:-1] * inputs)
            #out = 1.0 / (1.0 + np.exp(-weighted_sum))
            out = sigmoid(weighted_sum)
            #print("out=",out)
            neuron['out'] = out
            next_inputs.append(out)
        return next_inputs

class Network:
    def __init__(self, input_dim, hidden_dim, output_dim, init):
        self.hidden1 = Layer(input_dim, hidden_dim, init)
        self.hidden2 = Layer(hidden_dim, hidden_dim, init)
        self.output = Layer(hidden_dim, output_dim, init)

        self.layers = [self.hidden1, self.hidden2, self.output]


    def __getitem__(self, index):
        return self.layers[index]


    def forward(self, X):
        '''
        a forward pass through the network.
        '''
        inputs = X
        for layer in self.layers:
            next_inputs = layer.sum_and_activate(inputs)
            inputs = next_inputs
        return inputs

    def update_weights(self, training_input, gamma):
        '''
        use the gradients of each neuron to calculate the gradient with respect to the weights and then apply the update rule.
        '''
        # update rule is w = w - gamma*(grad_neuron * input)
        for i, layer in enumerate(self.layers):
            if i == 0:
                # inputs + a 1 for the bias term
                inputs = np.append(training_input, [1])
            else:
                # inputs + a 1 for the bias term
                inputs = np.array([n['out'] for n in self.layers[i - 1]] + [1])

            # update the weights at the current layer
            for neuron in layer:
                for j in range(len(inputs)):
                #print("neuron['grad']=",neuron['grad'])
                    neuron['weights'][j] -= gamma*neuron['grad']*inputs[j]



    def backprop(self, label):
        '''
        compute the gradient delta L / delta z_i for each neuron in the network.
        '''
        reversed_layers = self.layers[::-1]
        # first calculate the derivatives of loss w.r.t. each hidden state
        # Note: layers have been reversed so index 0 is actually last layer.
        for i, layer in enumerate(reversed_layers):
            #print("i=",i)

            # output neuron
            if i == 0:
                for neuron in layer:
                    neuron['grad'] = neuron['out'] - label


            # output neuron y = w^tx . That is, no activation.
            elif i == 1:
                for neuron in layer:
                    parent_grads = [n['grad'] * neuron['weights'][j] for j, n in enumerate(reversed_layers[i - 1])]
                    # add gradient info to network
                    neuron['grad'] = np.sum(parent_grads)

            # hidden states
            else:
                # current layer
                for neuron in layer:
                    # grad for current neuron is dL/dz^i-1 dz^i/dz^i
                    parent_grads = [n['grad'] * sigmoid_prime(n) * n['weights'][j] for j, n in enumerate(reversed_layers[i - 1])]
                    grad_sum = np.sum(parent_grads)
                    # add gradient info to network
                    neuron['grad'] = grad_sum

        # Update the weights using these gradients
        #self.update

    def predict(self, data):
        preds = []
        # Make predictions and calculate training and test error
        for instance in data:
            out = self.forward(instance)
            #print("out=",out)
            #pred = np.where(out[0] < .5, -1, 1)
            pred = np.where(out[0] < .5, 0, 1)
            preds.append(pred)
        return np.array(preds)


    def __str__(self):
        return '\n'.join([f'Layer {i}: {layer}' for i, layer in enumerate(self.layers)])
            

def sigmoid_prime(neuron):
    '''
    the partial derivative of the loss w.r.t.
    '''
    return neuron['out'] * (1.0 - neuron['out'])

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def prepare_data(path):
    '''
    Pre-process the dataset and return instances + labels
    '''
    with open(path, 'r') as f:
        features = []
        labels = []
        for line in f:
            cmpts = line.strip().split(',')
            features.append(cmpts[:-1])
            labels.append(cmpts[-1])
        # convert to floats
        features = np.array(features, dtype=float)
        labels = np.array(labels, dtype=float)

        #converted_labels = convert_labels(labels)


        #return features, converted_labels
        return features, labels

def convert_labels(y):
    '''
    convert labels to {1, -1}
    '''
    labels = np.where(y == 1, y, -1)
    return labels

def shuffle(X, y):
    '''
    randomly shuffle in unison the train instances and the labels.
    '''
    assert len(X) == len(y)
    rng = np.random.default_rng()
    perm = rng.permutation(len(X))
    shuffled_X = X[perm] 
    shuffled_y = y[perm]
    return X, y

def schedule(gamma, d, t):
    '''
    Return lr to be used for current epoch given a schedule.
    '''
    return gamma / (1 + (gamma / d)*t)



if __name__ == '__main__':

    # read in the dataset
    train_file = '../data/bank-note/train.csv'
    test_file = '../data/bank-note/test.csv'

    train_X, train_y = prepare_data(train_file)
    test_X, test_y = prepare_data(test_file)


    # for reproducibility
    np.random.seed(47)

    #initialize network
    nn = Network(4, args.hidden_dim, 1, args.init)
    #print("nn=",nn)
    #print('\n')

    print('############')
    print('Starting Training Loop')
    print(f'Running for {args.epochs} epochs')

    #SGD training loop
    for epoch in range(args.epochs):
        # randomly shuffle the data
        X, y = shuffle(train_X, train_y)
        # set the learning rate for this epoch
        gamma = schedule(args.gamma, args.d, epoch)
        #print("gamma=",gamma)
        epoch_loss = 0

        for instance, label in zip(X,y):
            #print("instance=",instance)
            #print("label=",label)
            # forward pass to get prediction
            pred = nn.forward(instance)
            #print("pred=",pred)

            # backpropagate the errors
            nn.backprop(label)

            #update the weights of the network
            nn.update_weights(instance, gamma)

            # Loss
            loss = (1/2)*(pred - label)**2 
            #print("loss=",loss)
            epoch_loss += loss
        print("epoch_loss=",epoch_loss)

    # Calculate train and test error
    print()
    print('RESULTS')
    train_preds = nn.predict(train_X)
    #print("train_preds=",train_preds)
    test_preds = nn.predict(test_X)

    train_error = len(np.where(train_preds != train_y)[0]) / len(train_X)
    print("train_error=",train_error)

    #print("test_y.shape=",test_y.shape)
    #print("test_preds.shape=",test_preds.shape)
    test_error = len(np.where(test_preds != test_y)[0]) / len(test_X)
    print("test_error=",test_error)

    print('############')
    print()

