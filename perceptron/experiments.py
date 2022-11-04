# Program to run the Perceptron Experiments from HW 3

import numpy as np
from perceptron import Standard_Perceptron, Voted_Perceptron, Averaged_Perceptron
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices = ['std', 'voted', 'avg', 'sklearn'], required=True)
parser.add_argument('--lr', type=float, required=True, default=.1)
parser.add_argument('--epochs', type=int, required=True, default=10)

args = parser.parse_args()

def prepare_data(path):
    '''
    Pre-process the bank-note data. Specific to this dataset.
    '''
    # load the data
    with open(path, 'r') as f:
        X = []
        y = []
        for line in f:
            cmpts = line.strip().split(',')
            X.append(cmpts[:-1])
            y.append(cmpts[-1])

    # convert to numpy and float
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    return X, y

def score(preds, labels):
    '''
    Compute average error on test set.
    '''
    # convert perceptron preds {-1, 1} back to {0,1} to match labels.
    preds = np.where(preds == -1, 0, 1)
    return np.sum(preds != labels) / len(preds)
    

if __name__ == '__main__':

    train_path = '../data/bank-note/train.csv'
    test_path = '../data/bank-note/test.csv'
    
    train_X, train_y = prepare_data(train_path)
    test_X, test_y = prepare_data(test_path)

    # Initialize a Perceptron
    if args.model == 'std':
        print()
        print('#######Standard Perceptron#########')
        print()

        model = Standard_Perceptron(epochs=args.epochs, lr=args.lr)
        print("model=",model)
        print("model.epochs=",model.epochs)
        print("model.lr=",model.lr)

        #model.train(train_X, train_y)

        #weight_vector = getattr(model, 'w')
        weight_vector = model.train(train_X, train_y)

        print("weight_vector=",weight_vector)

        # predict on training data
        preds_train = model.predict(train_X)

        preds_test = model.predict(test_X)

        # compute average training error
        avg_train_error = score(preds_train, train_y)
        avg_test_error = score(preds_test, test_y)

        print("avg_train_error=",avg_train_error)
        print("avg_test_error=",avg_test_error)


    if args.model == 'voted':
        print()
        print('#######Voted Perceptron#########')
        print()
        model = Voted_Perceptron(epochs=args.epochs, lr=args.lr)
        print("model=",model)
        print("model.epochs=",model.epochs)
        print("model.lr=",model.lr)

        # get classifiers and corresponding lifecycle counts
        w_hist, counts = model.train(train_X, train_y)

        w_plus_counts = [(w, c) for w,c in zip(w_hist, counts)]
        print("w_plus_counts=",w_plus_counts)

        # make predictions
        preds_train = model.predict(train_X)
        preds_test = model.predict(test_X)

        # compute average training error
        avg_train_error = score(preds_train, train_y)
        avg_test_error = score(preds_test, test_y)

        print("avg_train_error=",avg_train_error)
        print("avg_test_error=",avg_test_error)
        print()
        print('#####################')
        print()


    if args.model == 'avg':
        print()
        print('#######Averaged Perceptron#########')
        print()

        model = Averaged_Perceptron(epochs=args.epochs, lr=args.lr)
        print("model=",model)
        print("model.epochs=",model.epochs)
        print("model.lr=",model.lr)


        # get the averaged weight vector
        a = model.train(train_X, train_y)
        print("a=",a)

        # make predictions
        preds_train = model.predict(train_X)
        preds_test = model.predict(test_X)

        # compute average training error
        avg_train_error = score(preds_train, train_y)
        avg_test_error = score(preds_test, test_y)

        print("avg_train_error=",avg_train_error)
        print("avg_test_error=",avg_test_error)

        print()
        print('#####################')
        print()

