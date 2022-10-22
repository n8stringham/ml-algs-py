# This file implements batch and stochastic gradient descent for linear regression.

import argparse
import numpy as np
from numpy import linalg as LA
import random

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['bgd', 'sgd', 'analytic'])

args = parser.parse_args()

def lms_grad(x, y, w):
    '''
    return the gradient vector for least mean squares.
    '''
    # dimensions of the gradient vector
    d = x.shape[1]
    # calculate each element of the grad 
    grad_vec = [] 
    for j in range(d):
        diff = (y - x @ w) * x[:, j]
        # add element to grad
        grad_vec.append(-np.sum(diff))
    
    #print("grad_vec=",grad_vec)
    #print("np.array(grad_vec).dtype=",np.array(grad_vec).dtype)
    return np.array(grad_vec)

def lms_cost(x, y, w):
    '''
    calculate the mean square cost w.r.t. w
    '''
    # rows of x is the number of training examples
    m = x.shape[0]
    # sum over all training examples
    diff = (y - x @ w)**2
    #print("diff.shape=",diff.shape)

    res = np.sum(diff) / 2
    return res
    
def batch_gradient_descent(x, y, lr, thresh=10**-6, max_iters=500):
    '''
    Solve for the weights and bias term using batched gradient descent.
    '''
    # initialize weight vector - number of features + bias term
    w = np.zeros(x.shape[1])
    #grad_lms = 
    costs = []

    # record initial cost 
    costs.append(lms_cost(x, y, w))

    for i in range(max_iters):
        w0 = w
        # update weight vector
        w = w0 - lr*lms_grad(x, y, w)
        #print("w=",w)

        # record cost after update
        costs.append(lms_cost(x, y, w))

        # threshold check if converges quicker
        if LA.norm(w - w0) <= thresh:
            return w, costs
    return w, costs

def stochastic_gradient_descent(x, y, lr, max_iters=5, thresh=10**-6, seed=47):
    '''
    Solve for the weights and bias term using batched gradient descent.
    '''
    # initialize weight vector - number of features + bias term
    w = np.zeros(x.shape[1])
    #grad_lms = 
    costs = []

    # set seed for reproducible runs
    np.random.seed(seed)
    # number of total iterations over the training data
    for i in range(max_iters):
        # record initial cost
        costs.append(lms_cost(x, y, w))
        # select a random index in the training data
        rand_idxs = np.random.choice(x.shape[0], size=x.shape[0], replace=False)
        rand_idxs = range(x.shape[0])
        #print("rand_idxs=",rand_idxs)
        # update the gradient after each training example.
        for rand_idx in rand_idxs:
            w0 = w
            # update weight vector
            w = w0 - lr*lms_grad(x[rand_idx, :].reshape((1, -1)), y[rand_idx], w)
            #print("w=",w)
            # record cost after each update.
            costs.append(lms_cost(x, y, w))

            # threshold check if converges quicker
            if LA.norm(w - w0) <= thresh:
                # record final cost
                costs.append(lms_cost(x,y,w))
                return w, costs
    #costs.append(lms_cost(x,y,w))
    return w, costs

def cost_on_test(w):
    '''
    calculate the cost on the test data.
    '''
    # load the data
    x_test = []
    y_test = []
    with open('../data/concrete/test.csv', 'r') as f:
        for line in f:
            concrete = line.strip().split(',')
            x_test.append(concrete[:-1])
            y_test.append(concrete[-1])

    # turn into numpy arrays
    # add bias term column to x_train
    x_test = np.concatenate((np.ones(len(x_test))[:, np.newaxis], np.array(x_test, dtype=np.float64)), axis=1)
    y_test = np.array(y_test, dtype=np.float64)

    return lms_cost(x_test, y_test, w)

def analytic(x, y):
    '''
    compute the best weight vector using closed form analytic solution.
    '''
    X = x.T
    #print("y.shape=",y.shape)
    #print("X.shape=",X.shape)
    #print("x.shape=",x.shape)

    w_final = LA.inv(X@x) @ (X@y)
    return w_final



if __name__ == '__main__':
    # load the data
    x_train = []
    y_train = []
    #with open('../data/concrete/train.csv', 'r') as f:
    with open('table1.txt', 'r') as f:
        for line in f:
            concrete = line.strip().split(',')
            x_train.append(concrete[:-1])
            y_train.append(concrete[-1])

    # turn into numpy arrays
    # add bias term column to x_train
    x_train = np.concatenate((np.ones(len(x_train))[:, np.newaxis], np.array(x_train, dtype=np.float64)), axis=1)
    y_train = np.array(y_train, dtype=np.float64)
    #w0 = np.zeros(x_train.shape[1])
   # w0 = np.array([0, 0, 0, 0])

   # print("x_train=",x_train)
   # print("y_train=",y_train)

   # print("lms_grad(x_train, y_train, w0)=",lms_grad(x_train, y_train, w0))

   # print("analytic(x_train, y_train)=",analytic(x_train, y_train))
   # best_w = analytic(x_train, y_train)
   # w, costs = stochastic_gradient_descent(x_train, y_train, lr=.1, max_iters=1000)
   # #train_cost = lms_cost(x_train, y_train, best_w)
   # #print("train_cost=",train_cost)
   # #print("costs=",costs)
   # #print("w=",w)
   # asd

    # use specified optimization mode
    if args.mode == 'bgd':
        print('RUNNING BATCH GRADIENT DESCENT')
        w, costs = batch_gradient_descent(x_train, y_train, lr=.01, max_iters=100)

    if args.mode == 'sgd':
        print('RUNNING STOCHASTIC GRADIENT DESCENT')
        w, costs = stochastic_gradient_descent(x_train, y_train, lr=.005, max_iters=20)

    if args.mode == 'analytic':
        print()
        print('COMPUTING WEIGHTS ANALYTICALLY')
        final_w = analytic(x_train, y_train)
        print("final_w=",final_w)
        train_cost = lms_cost(x_train, y_train, final_w)
        print("train_cost=",train_cost)
        test_cost = cost_on_test(final_w)
        print("test_cost=",test_cost)


    else:
        # Print out the cost history and the final w
        #print("costs=",costs)
        final_train_cost = costs[-1]
        final_w = w
        print('Cost after each update')
        for c in costs:
            print("c=",c)

        print("final_train_cost=",final_train_cost)
        print("final_w=",final_w)
        # use the final w to compute cost on test data
        test_cost_start = cost_on_test(w0)
        test_cost_end = cost_on_test(w)
        print("test_cost_start=",test_cost_start)
        print("test_cost_end=",test_cost_end)
        print()

        # UNCOMMENT TO PLOT COST
        # plot the cost over time
        plt.plot(np.arange(len(costs)), costs)
        plt.xlabel('number of updates')
        plt.ylabel('lms cost')
        plt.savefig(f'optimize-{args.mode}')
        #plt.show()




