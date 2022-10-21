# Run experiments with Random Forests - Bagged Decision Trees where at each node we select from a random subset of the remaining attributes to split on.

from DecisionTree import Node, DecisionTree

import pickle
import argparse
from collections import Counter, defaultdict
import math
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Command Line Args
parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--test', required=True)
parser.add_argument('--T', required=True, type=int)

args=parser.parse_args()

# Functions for Processing and Training
def read_csv(path):
    '''
    Read csv file into a list of lists.
    '''
    with open(path, 'r') as f:
        #attr_vals = set()
        instances = [line.strip().split(',') for line in f]
        #labels = [i[-1] for i in instances]
    return instances

# #######
# Metrics
# #######
def entropy(S):
    '''
    Return the entropy of set of examples S.
    '''
    # group instances by label
    g = S.groupby(S.columns[-1])
    # size of each label
    terms = [len(g.get_group(label)) for label in g.groups.keys()] 
    num_instances = sum(terms)

    # entropy calculation
    H = sum(-(t/num_instances)*math.log((t/num_instances), 2) for t in terms)
#    print("terms=",terms)
#    print("num_instances=",num_instances)
#    print("H=",H)
    return H

# ########################
# Processing the input data 
###########################
def is_numeric(attr_vals):
    '''
    Determine if an attribute is numeric or not based on its values.
    '''
    # if all values can be cast to float, assume it is numeric feature.
    try:
        float_vals = [float(s) for s in attr_vals]
    except ValueError:
        return False
    return True

def get_attr_vals(df):
    '''
    Return a dictionary with the possible values for each attr sorted ascending.
    '''
    attr_vals = {}
    for a in df.columns:
        attr_vals[a] = sorted(list(df[a].unique()))
    return attr_vals

def numeric2categorical(df, fill_unk=False):
    '''
    Return a new df where the numeric attributes have been changed to a binary attribute 'greater' or 'less' depending on if val >= threshold. Take the threshold to be the median of the attribute values.
    '''
    new_df = df.copy()
    #Convert numerical values to categorical (set median value as the threshold and make two feature values less, greater)
    attr_vals = get_attr_vals(df)
    for k, v in attr_vals.items():
        # if v is numeric change corresponding col in df 
        if is_numeric(v):
            # adjust the possible values in attr_vals
            attr_vals[k] = ['greater', 'less']
            # first make sure corresponding col is numeric
            new_df[k] = pd.to_numeric(new_df[k])
            # find median and use as threshold
            thresh = statistics.median(new_df[k])
            #convert that col in the train_df to the new values
            new_df.loc[new_df[k] >= thresh, k] = 'greater'
            # everything left is <
            # janky I know, but I couldn't do the < comparison directly since string values in the column.
            new_df.loc[new_df[k] != 'greater', k] = 'less'
        
        # fill unk values with the majority value from training set.
        if fill_unk:
            if 'unknown' in attr_vals[k]:
                attr_vals[k].remove('unknown')
                # make sure we can't choose unk as majority val
                options = [o for o in new_df[k].value_counts().index if o != 'unknown']
                majority_val = options[0]
                #print("majority_val=",majority_val)
                new_df.loc[new_df[k] == 'unknown', k] = majority_val
    #print("attr_vals=",attr_vals)
    return new_df, attr_vals

# #######################
# Functions for Reporting

def print_results(avg_train_error, avg_test_error, errors, metric):
    '''
    Display the results.
    '''
    print(f'Results using {metric}')
    print()
    print('D | Train Error | Test Error')
    for i, e in enumerate(errors, start=1):
        print(f'{i}        {e[0]:<8}       {e[1]:<16}')

    print(f'average train error = {avg_train_error}')
    print(f'average test error = {avg_test_error}')
    print('\n')


def aggregate_preds(data, models):
    '''
    Bag the decision trees.
    '''
    #print("data=",data)
    # list of lists containing the votes for each model
    all_votes = []
    # for each model (with it's own alpha_t)
    for m in models:
        # the weighted vote for each data point for a given model
        votes = []
        # make model prediction
        preds = m.predict_all(data.values)
        #print("preds=",preds)
        for p in preds:
            # yes/no converted to 1/-1
            int_pred = 1 if p == 'yes' else -1
            # store weighted votes
            votes.append(int_pred)
        # add votes list 
        all_votes.append(votes)

    # convert to numpy
    all_votes = np.array(all_votes)
    #print("all_votes=",all_votes)
    final_preds = all_votes.sum(axis=0)
    #print("final_preds.shape=",final_preds.shape)

    # convert back to yes/no
    return ['yes' if pred >=0 else 'no' for pred in final_preds]

def compute_error(preds, labels):
    '''
    return the weighted/unweighted error for a set of predictions.
    '''
    #print("preds=",preds[:10])
    #print("labels[:10]=",labels[:10])

    incorrect = [p for p,l in zip(preds, labels) if p != l]
    error = len(incorrect) / len(preds)

    return error
        


def run_experiments(train_df, test_df, metric, T, fill_unk=False):
    '''
    Construct DecisionTrees for depths 1-6 using specified metric. Record the train and test error for each tree and compute the average prediction error. Return avg train error and avg test error
    '''
    
    # Pre-process to convert numeric to categorical
    # And decide how to handle UNK
    train_df, attr_vals = numeric2categorical(train_df, fill_unk)

    # only need the attr_vals from train_df
    test_df = numeric2categorical(test_df)[0]
    #print("attr_vals=",attr_vals)

    # store all of the models
    models = []
    # Bagging Loop
    # Train T classifiers
    train_errors = []
    test_errors = []
    for i in range(T):
        # select a random sample from the training set
        sample = train_df.sample(len(train_df), replace=True)

        #initialize model
        model = DecisionTree(metric, rf=True, rf_sample_size=size)

        model.train(sample, attr_vals)
        #print("model.root=",model.root)

        # store model for use to predict on test data
        models.append(model)

        # Compute the Final Hypothesis Train
        h_final_train = aggregate_preds(train_df, models)
        #print("h_final=",h_final)
        train_error = compute_error(h_final_train, train_df.iloc[:, -1].to_list())
        print("train_error=",train_error)
        train_errors.append(train_error)

        # Compute the final hypothesis for Test
        h_final_test = aggregate_preds(test_df, models)
        test_error = compute_error(h_final_test, test_df.iloc[:, -1].to_list())
        print("test_error=",test_error)
        test_errors.append(test_error)

    return train_errors, test_errors


# Running the Experiments
if __name__ == '__main__':
    print(f"Running Bagging Experiments for HW2 on Dataset at {'/'.join(args.train.split('/')[:3])}")
    
    train_instances = read_csv(args.train)
    test_instances = read_csv(args.test)

    # put in a df
    train_df = pd.DataFrame(train_instances)
    test_df = pd.DataFrame(test_instances)

   # FIXME uncomment to run experiment 2D
    # vary the random subset size
   # for size in [2, 4, 6]:
   #     print('RUNNING BAGGING EXPERIMENT 2D')
   #     print('This could take hours')
   #     res = run_experiments(train_df, test_df, entropy, args.T, fill_unk=False)

   #     xs = range(1, args.T + 1)
   #     plt.plot(xs, res[0], label='train error')
   #     plt.plot(xs, res[1], label='test error')
   #     plt.legend()
   #     plt.title(f'RF with sample size {size}')
   #     plt.savefig(f'2D-{size}.pdf')
   # asd
        #plt.show()


    # 2E
    # list of list of models. each list represents a bagged decision tree composed of trees. bagged models is the total number of ensemble models. 
    print('RUNNING BAGGING EXPERIMENT 2C')
    print('This could take hours')
    bagged_models = []
    for i in range(50):
        # sample 1000 training instances
        sample = train_df.sample(n=1000)
        
        # Pre-process to convert numeric to categorical
        # And decide how to handle UNK
        train_df, attr_vals = numeric2categorical(train_df, fill_unk=False)

        # only need the attr_vals from train_df
        #test_df = numeric2categorical(test_df)[0]
        #print("attr_vals=",attr_vals)

        # store all of the models
        models = []

        # Bagging Loop
        # Train T classifiers
        #train_errors = []
        #test_errors = []
        for j in range(30):
            if j % 10 == 0:
                print(f'training tree {j}')

            #initialize model
            model = DecisionTree(entropy, rf=True, rf_sample_size=2)

            # train 500 bagged decision trees.
            model.train(sample, attr_vals)
            #print("model.root=",model.root)

            # store model for use to predict on test data
            models.append(model)

        # save the bagged decision tree
        with open(f'bagged_trees/rf-bagged-{i}.pkl', 'wb') as f:
            pickle.dump(models, f)

        #with open(f'bagged_trees/bagged-{0}.pkl', 'rb') as f:
            #ms = pickle.load(f)

        print('iterateion ', i)
        
        # add the list of models to bagged models
        #bagged_models.append(models)

        ## Compute the Final Hypothesis Train
        #h_final_train = aggregate_preds(train_df, models)
        ##print("h_final=",h_final)
        #train_error = compute_error(h_final_train, train_df.iloc[:, -1].to_list())
        #print("train_error=",train_error)
        #train_errors.append(train_error)

        ## Compute the final hypothesis for Test
        #h_final_test = aggregate_preds(test_df, models)
        #test_error = compute_error(h_final_test, test_df.iloc[:, -1].to_list())
        #print("test_error=",test_error)
        #test_errors.append(test_error)



