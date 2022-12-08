# this script trains an svm on the bank-note.zip dataset and runs the experiments for HW4

import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

from svm import SVM, DualSVM

parser = argparse.ArgumentParser()
parser.add_argument('--epochs')
parser.add_argument('--C', type=float)
parser.add_argument('--lr', type=float)
parser.add_argument('--mode', choices=['primal', 'dual'])
parser.add_argument('--lr-schedule', choices=['a', 'b', 'diff'])
parser.add_argument('--lr-a', type=float)
parser.add_argument('--diff', action='store_true')

parser.add_argument('--dual', action='store_true')
parser.add_argument('--gamma', type=float)
parser.add_argument('--kernel', choices=['linear', 'gaussian'])

parser.add_argument('--test', action='store_true')
parser.add_argument('--overlap', action='store_true')

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

def simulate(X, y, lr, a, c, schedule, num_trials):
    '''
    Run monte carlo simulations with a specific set of params.
    '''
    running_sum = 0
    for _ in range(num_trials):
        model = SVM(lr=lr, a=a, C=c, schedule=schedule)
        model.train(X, y)
        preds = model.predict(X)
        error = model.error(preds, y)
        print("error=",error)
        running_sum += error
    avg_error = running_sum / num_trials
    print("avg_error=",avg_error)


# read in the dataset
train_file = '../data/bank-note/train.csv'
test_file = '../data/bank-note/test.csv'

train_X, train_y = prepare_data(train_file)
test_X, test_y = prepare_data(test_file)

#simulate(train_X, train_y, args.lr, args.lr_a, args.C, args.lr_schedule, 10)

#print("train_X=",train_X)
#print("train_y=",train_y)

# Experiment Hyperparam ranges
#Cs = [100/873, 500/873, 700/873]


#
if args.dual:
    model = DualSVM(C=args.C, kernel=args.kernel, gamma=args.gamma)
    # Train the model
    model.train(train_X, train_y)

    print('Model Hyperparameters')
    print("args.kernel=",args.kernel)
    print("args.C=",args.C)
    print("args.gamma=",args.gamma)
    print()
    train_preds = model.predict(train_X)
    train_error = model.error(train_preds, train_y)
    test_preds = model.predict(test_X)
    test_error = model.error(test_preds, test_y)
    print("train_error=",train_error)
    print("test_error=",test_error)

    print("model.w=",model.w)
    print("model.b=",model.b)

    if args.overlap:
        model1 = DualSVM(C=args.C, kernel=args.kernel, gamma=.1)
        model2 = DualSVM(C=args.C, kernel=args.kernel, gamma=.5)
        model3 = DualSVM(C=args.C, kernel=args.kernel, gamma=1)
        model4 = DualSVM(C=args.C, kernel=args.kernel, gamma=5)
        model5 = DualSVM(C=args.C, kernel=args.kernel, gamma=100)

        support_vectors = []
        for m in [model1, model2, model3, model4, model5]:
            m.train(train_X, train_y)
            m.recover_weights()
            #support_vectors.append(m.svs)
            support_vectors.append(m.sv_idxs)
            print("len(m.svs)=",len(m.svs))

        for i in range(len(support_vectors) - 1):
            #intersect = np.intersect1d(support_vectors[i], support_vectors[i+1])
            #print("support_vectors[i]=",support_vectors[i])
            intersect = np.logical_and(support_vectors[i], support_vectors[i+1])
            
            print("len(train_X[intersect])=",len(train_X[intersect]))

else:
# find diff in model params
    if args.diff:
        model_a = SVM(lr=args.lr, a=args.lr_a, C=args.C, schedule='a')
        model_b = SVM(lr=args.lr, a=args.lr_a, C=args.C, schedule='b')

        # stats for model_a
        objectives_a, steps_a = model_a.train(train_X, train_y)
        train_preds_a = model_a.predict(train_X)
        train_error_a = model_a.error(train_preds_a, train_y)
        test_preds_a = model_a.predict(test_X)
        test_error_a = model_a.error(test_preds_a, test_y)
        print("train_error_a=",train_error_a)
        print("test_error_a=",test_error_a)

        # stats for model_b
        objectives_b, steps_b = model_b.train(train_X, train_y)
        train_preds_b = model_b.predict(train_X)
        train_error_b = model_b.error(train_preds_b, train_y)
        test_preds_b = model_b.predict(test_X)
        test_error_b = model_b.error(test_preds_b, test_y)
        print("train_error_b=",train_error_b)
        print("test_error_b=",test_error_b)

        diff_train_error = train_error_a - train_error_b
        diff_test_error = test_error_a - test_error_b
        diff_weights = model_a.w - model_b.w
        print("diff_train_error=",diff_train_error)
        print("diff_test_error=",diff_test_error)
        print("diff_weights=",diff_weights)

        print("model_a.w=",model_a.w)
        print("model_b.w=",model_b.w)

    else:
        model = SVM(lr=args.lr, a=args.lr_a, C=args.C, schedule=args.lr_schedule)
        print('Model Hyperparameters')
        print("args.lr=",args.lr)
        print("args.lr_a=",args.lr_a)
        print("args.C=",args.C)
        print("args.lr_schedule=",args.lr_schedule)
        print()

        ### train the model
        objectives, steps = model.train(train_X, train_y)
        train_preds = model.predict(train_X)
        train_error = model.error(train_preds, train_y)
        test_preds = model.predict(test_X)
        test_error = model.error(test_preds, test_y)
        print("train_error=",train_error)
        print("test_error=",test_error)
        print("model.w=",model.w)



## plot the objective function curve
#plt.plot(steps, objectives)
#plt.title('SVM Objective Function Curve')
#plt.xlabel('steps')
#plt.ylabel('objective value')
#
#plt.show()
