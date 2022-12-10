# Run the experiments for HW5
#
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F

from network_bonus import Model

import argparse
import numpy as np
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['python', 'torch'])
parser.add_argument('--num_layers', required=True, type=int)
parser.add_argument('--width', required=True, type=int)
parser.add_argument('--initialization', choices= ['He','Xavier'], required=True)
parser.add_argument('--activation', choices=['tanh', 'ReLU'], required=True)

args = parser.parse_args()



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

        converted_labels = convert_labels(labels)


        return features, converted_labels

def convert_labels(y):
    '''
    convert labels to {1, -1}
    '''
    labels = np.where(y == 1, y, -1)
    return labels

class BankNoteDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_loop(dataloader, model, optimizer):
    '''
    run the training loop for some set of params.
    '''
    # training loop
    for i in range(15):
        #for instance, label in zip(train_X, train_y):
        for batch, (X, y) in enumerate(train_dataloader):
            # forward pass
            pred = model(X.float())
            #print("pred=",pred)
            # computing square loss
            loss = 1/2*(pred - y)**2
            #print("loss=",loss.item())

            # Backpropagation
            # zero out the gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    return model
    

def test_loop(dataloader, model, activation):
    '''
    make predictions for the test set and return error
    '''
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.float())
            if activation == 'ReLU':
                pred = torch.where(pred > .5, 1, -1)
            elif activation == 'tanh':
                pred = torch.where(pred > 0, 1, -1)
            test_loss += 1/2*(pred - y)**2
            #print("test_loss=",test_loss)

            correct += (y == pred)

    error = 1 - (correct / len(dataloader.dataset))
    return error.item()


if __name__ == '__main__':

    # read in the dataset
    train_file = '../data/bank-note/train.csv'
    test_file = '../data/bank-note/test.csv'

    train_X, train_y = prepare_data(train_file)
    test_X, test_y = prepare_data(test_file)


    # Part 2E
    if args.mode == 'torch':
        # use pytorch Datasets and Dataloaders
        train_dataset = BankNoteDataset(train_X, train_y)
        test_dataset = BankNoteDataset(test_X, test_y)

        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        # set the model and optimizer to use
        # Run the experiments with different weight
        if args.activation == 'ReLU':
            activation = F.relu
        if args.activation == 'tanh':
            activation = torch.tanh

        model = Model(input_dim=4, hidden_dim=args.width, output_dim=1, activation=activation, initialization=args.initialization, num_layers=args.num_layers)
        
        optimizer = torch.optim.Adam(model.parameters())

        # train the model and report train and test error
        trained = train_loop(train_dataloader, model, optimizer)
        train_error = test_loop(train_dataloader, trained, args.activation)
        test_error = test_loop(test_dataloader, trained, args.activation)
        print('Model trianed with the following hyperparams')
        print(f'Activation: {args.activation}\nInitialization: {args.initialization}\nWidth: {args.width}\nDepth: {args.num_layers}\n')
        print("train_error=",train_error)
        print("test_error=",test_error)

        print('\n###############################\n')

