# this program implements the AdaBoost algorithm for use with decision stumps. This requires us to modify Information Gain to use weights.

from DecisionStump import Node, DecisionStump 

import argparse
from collections import Counter, defaultdict
import math
import statistics
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pickle

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
    #print("models=",models)
    #print("data=",data)
    # list of lists containing the votes for each model
    all_votes = []
    # for each model (with it's own alpha_t)
    for m, a in zip(models, alphas):
        # the weighted vote for each data point for a given model
        votes = []
        # make model prediction
        preds = m.predict_all(data.values)
        #print("preds[52:67]=",preds[52:67])
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
    #print("all_votes=",all_votes)
    final_preds = all_votes.sum(axis=0)
    #print("final_preds=",final_preds)
    #print("final_preds.shape=",final_preds.shape)

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
        


def run_experiments(train_df, test_df, metric, fill_unk=False, first_exp=True):
    '''
    Construct DecisionTrees for depths 1-6 using specified metric. Record the train and test error for each tree and compute the average prediction error. Return avg train error and avg test error
    '''
    
    # construct DecisionTree from train dataframe
    #model.train(train_df)
    #######
    # Pre-process to convert numeric to categorical
    # And decide how to handle UNK
    train_df, attr_vals = numeric2categorical(train_df, fill_unk)
    test_df = numeric2categorical(test_df, fill_unk)[0]

    print("test_df=",test_df)
    # Insert weights column initialized as uniform dist
    train_initial_weights = [1/len(train_df)]*len(train_df)
    train_df.insert(0, 'weights', train_initial_weights)

    # test df needs initial weights, but they won't be used
    test_df.insert(0, 'weights', train_initial_weights)


    # ######################
    # train the decision stump
    # ######################

    # each row is an array containing alpha_t * h_t for each data point x
    if first_exp:
        print('Training Adaboost models with decision stumps')
        print('Will compute errors vs T')
        models = []
        alphas = []
        #T = 5
        # Train T models, using different weights each time.
        for i in range(args.T):
            #initialize model
            model = DecisionStump(metric)

            model.train(train_df, attr_vals)
            print("model.root=",model.root)

            # store model for use to predict on test data
            models.append(model)

            preds = model.predict_all(train_df.values)
            print("preds[:10]=",preds[:10])
            labels = train_df.iloc[:, -1].to_list()
            weighted_error = compute_error(preds, labels, weighted=True)
            print("weighted_error=",weighted_error)

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
            #print("sum(new_weights)=",sum(new_weights))
            #print("new_weights[:10]=",new_weights[:10])

            # Update weights column in dataframe
            train_df['weights'] = new_weights
            #print('df updated with new weights')
            #print("train_df.head()=",train_df.head())

        # Save the models
        with open('adaboost_ensembles/models.pkl', 'wb') as f:
            pickle.dump(models, f)
        # Save the alphas
        with open('adaboost_ensembles/alphas.pkl', 'wb') as f:
            pickle.dump(alphas, f)

      #  # Load the models
      #  with open('adaboost_ensembles/models.pkl', 'rb') as f:
      #      models = pickle.load(f)


        # Predictions for ensemblse of sizes 1-500
        ensemble_train_errors = []
        ensemble_test_errors = []
        for size in range(1, len(models) + 1):
            #print("size=",size)
            #print("models[:size]=",models[:size])
            # Compute the Final Hypothesis for each data point Train Error
            h_final_train = final_preds(train_df, models[:size], alphas[:size])
            #print("h_final=",h_final)
            train_error = compute_error(h_final_train, train_df.iloc[:, -1].to_list())
            print("train_error=",train_error)
            ensemble_train_errors.append(train_error)
            # Compute the Final Hypothesis for each data point Test Error
            h_final_test = final_preds(test_df, models[:size], alphas[:size])
            #print("h_final=",h_final)
            test_error = compute_error(h_final_test, test_df.iloc[:, -1].to_list())
            print("test_error=",test_error)
            ensemble_test_errors.append(test_error)

        # plot the figure
        xs = range(1, len(ensemble_train_errors) + 1)
        plt.plot(xs, ensemble_train_errors, label='adaboost train error')
        plt.plot(xs, ensemble_test_errors, label='adaboost test error')
        plt.legend()
        plt.title('adaboost errors for different values of T')
        #plt.show()
        plt.savefig(f'2A-1.pdf')

    # Running the 2nd experiment
    else:
        print('Getting Errors for each decision Stump')
        with open('adaboost_ensembles/models.pkl', 'rb') as f:
            models = pickle.load(f)


        # calculate train and test error for each model
        stump_train_errors = []
        stump_test_errors = []
        for i, m in enumerate(models):
            train_preds = [p[0] for p in m.predict_all(train_df.values)]
            train_labels = train_df.iloc[:, -1].to_list()
            test_preds = [p[0] for p in m.predict_all(test_df.values)]

            train_error = compute_error(train_preds, train_df.iloc[:, -1].to_list())
            print("train_error=",train_error)
            test_error = compute_error(test_preds, test_df.iloc[:, -1].to_list())
            print("test_error=",test_error)
            stump_train_errors.append(train_error)
            stump_test_errors.append(test_error)

            
        # plot the figure
        xs = range(1, len(stump_train_errors) + 1)
        plt.plot(xs, stump_train_errors, label='DecisionStump train error')
        plt.plot(xs, stump_test_errors, label='DecisionStump test error')
        plt.xlabel('iteration')
        plt.legend()
        plt.title('Errors for Decision Stumps learned at each iteration')
        plt.savefig(f'2A-2.pdf')


# Running the Experiments
if __name__ == '__main__':
    print(f"Running Experiments for HW1 on Dataset at {'/'.join(args.train.split('/')[:3])}")
    
    train_instances = read_csv(args.train)
    test_instances = read_csv(args.test)

    # put in a df
    train_df = pd.DataFrame(train_instances)
    test_df = pd.DataFrame(test_instances)

    first_exp = run_experiments(train_df, test_df, entropy_with_weights, fill_unk=False, first_exp=True)

    second_exp = run_experiments(train_df, test_df, entropy_with_weights, fill_unk=False, first_exp=False)


    # print the tree
    #print("model.root=",model.root)
