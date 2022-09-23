'''
This file is used to run experiments with a DecisionTree.
'''


from DecisionTree import Node, DecisionTree 

import argparse
from collections import Counter, defaultdict
import math
import statistics
import pandas as pd

# Command Line Args
parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
#parser.add_argument('--attrs', required=True)
parser.add_argument('--test', required=True)
#parser.add_argument('--metric', choices=['IG', 'ME', 'GI'], required=True)

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

def majority_error(S):
    '''
    Return the majority error of set S.
    '''
    #print("S=",S)
    majority_count = S.iloc[:,S.columns[-1]].value_counts()[0]
    #print("majority_count=",majority_count)
    size_S = len(S)
    #print("size_S=",size_S)
    return (size_S - majority_count) / size_S

def gini(S):
    '''
    Return the Gini impurity of a set S
    '''
    #g = S.groupby(S.columns[-1])
    counts = S.iloc[:,S.columns[-1]].value_counts()
    #print("counts=",counts)
    size_S = len(S)
    #print("size_S=",size_S)
    sum_squared_ps = sum((p/size_S)**2 for p in counts)
    #print("sum_squared_ps=",sum_squared_ps)
    return 1 - sum_squared_ps

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

def metric_str(metric):
    '''
    Return the string version of the metric.
    '''
    if metric == gini:
        return 'Gini'
    if metric == entropy:
        return 'IG'
    if metric == majority_error:
        return 'ME'

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

def make_table(avg_train_error, avg_test_error, errors, metric, fill_unk):
    '''
    Turn the experiment results into a LaTex table
    '''
    #FIXME: Implement this when doing the write-up
    


def run_experiments(train_df, test_df, metric, max_depth, fill_unk=False):
    '''
    Construct DecisionTrees for depths 1-6 using specified metric. Record the train and test error for each tree and compute the average prediction error. Return avg train error and avg test error
    '''
    errors = []
    #for d in range(1,7):
    for d in range(1,max_depth+1):
        #initialize model
        model = DecisionTree(metric, depth_limit=d)
        
        # construct DecisionTree from train dataframe
        #model.train(train_df)
        #######
        # Pre-process to convert numeric to categorical
        # And decide how to handle UNK
        train_df, attr_vals = numeric2categorical(train_df, fill_unk)
        # only need the attr_vals from train_df
        test_df = numeric2categorical(test_df, fill_unk)[0]
        #print("attr_vals=",attr_vals)

        model.train(train_df, attr_vals)

        # train and test error w.r.t depth d
        error_d = [] 
        for dataset in [train_df, test_df]:
            labels = dataset.iloc[:,-1].to_list()

            #print("dataset.values=",dataset.values)

            # make predictions on dataset
            preds = model.predict_all(dataset.values)

            # compute the train and test errors
            incorrect = [p for p,l in zip(preds,labels) if p != l]
            #print("len(incorrect)=",len(incorrect))
            error = len(incorrect)/len(preds)
            #accuracy = 1 - error 
            #print("error=",error)
            error_d.append(error) #print("error_d=",error_d)
        # add the list [train_error, test error] to full errors list
        errors.append(error_d)
    # compute average errors for train and test
    avg_train_error = sum(e[0] for e in errors) / len(errors)
    avg_test_error = sum(e[1] for e in errors) / len(errors)



    # visualize the full decision tree
    #print("model.root=",model.root)

    return avg_train_error, avg_test_error, errors


# Running the Experiments
if __name__ == '__main__':
    print(f"Running Experiments for HW1 on Dataset at {'/'.join(args.train.split('/')[:3])}")
    
    train_instances = read_csv(args.train)
    test_instances = read_csv(args.test)

    # put in a df
    train_df = pd.DataFrame(train_instances)
    test_df = pd.DataFrame(test_instances)

    dataset = '/'.join(args.train.split('/')[:3])

    # run experiments
    if dataset == '../data/car-data':
        print('-----------------')
        print('Question 2a, 2b')
        print('-----------------')
        IG_scores = run_experiments(train_df, test_df, entropy, 6)
        ME_scores = run_experiments(train_df, test_df, majority_error, 6)
        GI_scores = run_experiments(train_df, test_df, gini, 6)

        print_results(*IG_scores, 'IG')
        print_results(*ME_scores, 'ME')
        print_results(*GI_scores, 'GI')

    elif dataset == '../data/bank-data':
        print('-----------------')
        print('Question 3a, 3b')
        print('This may take some time...')
        print('-----------------')
        for v in [False, True]:
            IG_scores = run_experiments(train_df, test_df, entropy, 16, fill_unk=v)
            ME_scores = run_experiments(train_df, test_df, majority_error, 16, fill_unk=v)
            GI_scores = run_experiments(train_df, test_df, gini, 16, fill_unk=v)

            print(f'unknown values treated as missing={v}')
            print_results(*IG_scores, 'IG')
            print_results(*ME_scores, 'ME')
            print_results(*GI_scores, 'GI')
            print('#######################')

    # print the tree
    #print("model.root=",model.root)
