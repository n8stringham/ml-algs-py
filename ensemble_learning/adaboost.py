# this program implements the AdaBoost algorithm for use with decision stumps. This requires us to modify Information Gain to use weights.

#import decision_tree.DecisionTree as DT


#Maybe just copy over code so we have access to DecisionTree or change the path...

# Or figure out how to package the decision_tree folder and import it.
#

# Algorithm
# Construct T Weak Learners
# Initially, the weights are from the uniform distribution.
# i.e. each example has importancde 1/m for m training examples.
#
# After each round, we want to get a learner that can correctly predict things that were wrong in the previous round.
#
# We create new weight D_t+1(i) = D_t / Z_t * exp(-alpha*y_i*h_i)
#
# Z_t is all of the weights
# alpha_t = 1/2 ln (1-epsilon_t/epsilon_t)

from DecisionStump import Node, DecisionStump 

import argparse
from collections import Counter, defaultdict
import math
import statistics
import pandas as pd
import numpy as np

# Command Line Args
parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--test', required=True)

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

def entropy_with_weights(S):
    '''
    Return the entropy of set of examples S. This function allows us to provide weights for each training example--treating them as fractional examples--for compatibility with AdaBoost.
    '''
    # group instances by label
    g = S.groupby(S.columns[-1])
    # size of each label
    terms = [g.get_group(label)['weights'].sum() for label in g.groups.keys()] 
    #num_instances = sum(terms)

    # entropy calculation
    H = sum(-t*math.log(t, 2) for t in terms)
    #print("H=",H)
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


def compute_alpha(weighted_error):
    '''
    compute the voting parameter for a weak learner.
    '''
    return (1/2)*math.log((1 - weighted_error)/weighted_error)


def final_preds(data, models, alphas):
    '''
    Ensemble the AdaBoost Models and return final pred
    '''
    print("data=",data)
    # list of lists containing the votes for each model
    all_votes = []
    # for each model (with it's own alpha_t)
    for m, a in zip(models, alphas):
        # the weighted vote for each data point for a given model
        votes = []
        # make model prediction
        preds = m.predict_all(data.values)
        for p in preds:
            # yes/no converted to 1/-1
            int_pred = 1 if p[0] == 'yes' else -1
            vote = a * int_pred
            # store weighted votes
            votes.append(vote)
        # add votes list 
        all_votes.append(votes)

    # convert to numpy
    all_votes = np.array(all_votes)
    print("all_votes=",all_votes)
    final_preds = all_votes.sum(axis=0)
    print("final_preds.shape=",final_preds.shape)

    # convert back to yes/no
    return ['yes' if pred >=0 else 'no' for pred in final_preds]

def compute_error(preds, labels, weighted=False):
    '''
    return the weighted/unweighted error for a set of predictions.
    '''
    #print("preds=",preds[:10])
    #print("labels[:10]=",labels[:10])

    if weighted:
        # calculate the weighted error
        incorrect = [p for p,l in zip(preds,labels) if p[0] != l]
        print("len(incorrect)=",len(incorrect))
        # sum up the weights for all of the incorrect answers
        error = sum([i[1] for i in incorrect])
    else:
        incorrect = [p for p,l in zip(preds, labels) if p != l]
        error = len(incorrect) / len(preds)

    return error
        


def run_experiments(train_df, test_df, metric, fill_unk=False):
    '''
    Construct DecisionTrees for depths 1-6 using specified metric. Record the train and test error for each tree and compute the average prediction error. Return avg train error and avg test error
    '''
    
    # construct DecisionTree from train dataframe
    #model.train(train_df)
    #######
    # Pre-process to convert numeric to categorical
    # And decide how to handle UNK
    train_df, attr_vals = numeric2categorical(train_df, fill_unk)

    # Insert weights column initialized as uniform dist
    train_initial_weights = [1/len(train_df)]*len(train_df)
    train_df.insert(0, 'weights', train_initial_weights)


    # only need the attr_vals from train_df
    #test_df = numeric2categorical(test_df)[0]
    #print("attr_vals=",attr_vals)

    # ######################
    # train the decision stump
    # ######################

    # each row is an array containing alpha_t * h_t for each data point x
    models = []
    alphas = []
    T = 500 
    # Train T models, using different weights each time.
    for i in range(T):
        #initialize model
        model = DecisionStump(metric)

        model.train(train_df, attr_vals)
        print("model.root=",model.root)

        # store model for use to predict on test data
        models.append(model)

        preds = model.predict_all(train_df.values)
        labels = train_df.iloc[:, -1].to_list()
        weighted_error = compute_error(preds, labels, weighted=True)
       # # calculate the weighted error
       # labels = train_df.iloc[:, -1].to_list()
       # preds = model.predict_all(train_df.values)
       # incorrect = [p for p,l in zip(preds,labels) if p[0] != l]
       # # sum up the weights for all of the incorrect answers
       # weighted_error = sum([i[1] for i in incorrect])
       # #print("weighted_error=",weighted_error)

        # Compute the alpha value for this learner
        alpha_t = compute_alpha(weighted_error)
        print("alpha_t=",alpha_t)
        alphas.append(alpha_t)

        # update weights for training examples
        new_weights = []
        # alpha_t * h_t for each data point
        votes = [] 
        incorrect = 0
        # calculate all of the new weights
        for p,l in zip(preds,labels):
            # this is same as y_i*h_t(x_i)
            indicator = 1 if p[0] == l else -1 
            # add unnormalized new weight
            new_weights.append(p[1]*math.exp(-alpha_t*indicator))


        #normalize new_weights
        Z = sum(new_weights)
        #print("new_weights[:10]=",new_weights[:10])
        #print("Z=",Z)
        new_weights = [w/Z for w in new_weights]
        print("sum(new_weights)=",sum(new_weights))
        #print("new_weights[:10]=",new_weights[:10])

        # Update weights column in dataframe
        train_df['weights'] = new_weights
        print('df updated with new weights')
        print("train_df.head()=",train_df.head())


    # Compute the Final Hypothesis for each data point Train Error
    h_final = final_preds(train_df, models, alphas)
    #print("h_final=",h_final)
    train_error = compute_error(h_final, train_df.iloc[:, -1].to_list())
    print("train_error=",train_error)
    asd

    

    # Score the final preds

    # Testing
    # 1. Make predictions on the test data using each model
    asd

    # Old Code
    #for dataset in [train_df, test_df]:
    #    labels = dataset.iloc[:,-1].to_list()

    #    #print("dataset.values=",dataset.values)

    #    # make predictions on dataset
    #    preds = model.predict_all(dataset.values)

    #    # compute the train and test errors
    #    incorrect = [p for p,l in zip(preds,labels) if p != l]
    #    #print("len(incorrect)=",len(incorrect))
    #    error = len(incorrect)/len(preds)
    #    #accuracy = 1 - error 
    #    #print("error=",error)
    #    error_d.append(error) #print("error_d=",error_d)
    ## add the list [train_error, test error] to full errors list
    #errors.append(error_d)
    ## compute average errors for train and test
    #avg_train_error = sum(e[0] for e in errors) / len(errors)
    #avg_test_error = sum(e[1] for e in errors) / len(errors)



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

    IG_scores = run_experiments(train_df, test_df, entropy_with_weights, fill_unk=False)


    # print the tree
    #print("model.root=",model.root)
