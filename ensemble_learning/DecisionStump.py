'''
This file implements the Node and DecisionStump classes.
'''
import argparse
from collections import Counter, defaultdict
import math
import statistics
import pandas as pd

class Node():
    '''
    A node can be of two types -- Decision or Leaf. 
    '''
    def __init__(self, attribute=None, children=None, value=None, branch=None):
        '''
        Create a Decision or Leaf Node.

        children -- list of Node objects
        '''

        # only decision nodes have an attribute
        self.attribute = attribute
        # only decision nodes have children
        self.children = children
        # only leaf nodes have a value
        self.value = value
        # marks the attribute value that led to this node
        self.branch = branch

    def __str__(self):
        '''
        Visualize a node.
        '''

        ret = '('
        if self.children:
            if self.branch:
                ret += str(self.branch)
            ret += str(self.attribute)
            ret += ' - '
            for c in self.children:
                ret += str(c)
                ret += ' '
            ret += '- '
        else:
            #ret += ' '
            ret += str(self.value)
            #ret += ' '
        ret += ')'
        return ret


class DecisionStump():
    def __init__(self, metric=None, depth_limit=0):
        assert metric is not None
        self.metric = metric
        self.depth_limit = depth_limit
        self.attr_vals = None

        # start with an empty tree
        self.root = None

    def __str__(self):
        '''
        Vizualize the Tree.
        '''
        return str(self.root)

    def grow(self, S, attributes, depth):
        '''
        Following ID3, recursively add nodes to the decision tree, returning the root.
        '''
        # only continue to grow if we haven't reached depth limit.
        #print("depth=",depth)
        if depth <= self.depth_limit:

            # all labels same -> create leaf node or no more attributes to split.
            #if self.metric(S) == 0 or len(attributes) == 0:
            if self.metric(S) == 0 or len(attributes) == 0:
                return Node(value=majority_label(S))

            # find the best attribute to split on
            best_attr, best_gain, branches = self.best_split(S, attributes)
            # recur on each of the new subsets
            children = []
            #print("attributes=",attributes)
            new_attrs = [a for a in attributes if a != best_attr]

            # consider each possible branch value
            for b in self.attr_vals[best_attr]:
                #print("self.attr_vals[best_attr]=",self.attr_vals[best_attr])
                # if there are still training instances, we can attempt to grow another sub-tree below
                if b in branches.groups.keys():
                    subset = branches.get_group(b)
                    #print("b=",b)
                    #print("subset=",subset)
                    #print("len(subset)=",len(subset))
                #if len(subset) > 0:
                    child = self.grow(subset, new_attrs, depth+1)
                # empty branch -> create leaf node with majority label
                else:
                    #print("S=",S)
                    child = Node(value=majority_label(S))
                    #print("child=",child)
                children.append(child)
            return Node(best_attr, children)
        return Node(value=majority_label(S))



    def best_split(self, S, attributes):
        '''
        Find the best attribute to split on.
        '''
        # for each of the attributes we need to calculate IG
        # Then we pick whichever attribute has the highest IG
        # If all labels the same, then no further split

        # calculate the purity of the current dataset S
        # using self.metric measure.
        purity_S = self.metric(S)
        #print("purity_S=",purity_S)

        # initialize best gain
        best_gain = -1
        best_attr = None
        best_subsets = None
        for a in attributes:
            gain_a, subsets_a = self.information_gain(S, purity_S, a)
            best_gain = max(best_gain, gain_a)
            if best_gain == gain_a:
                best_attr = a
                best_subsets = subsets_a
        print("best_attr=",best_attr)
        print("best_gain=",best_gain)
        print("best_subsets=",best_subsets)
        return best_attr, best_gain, best_subsets

    def information_gain(self, S, purity_S, attribute):
        '''
        Calculate the information gain from splitting on an attribute.
        '''
        #print('\n')
        #print(f'calculating IG for {attribute}')
        #print('\n')
        # get the dataframes groupedby attribute
        subsets = split_data(S, attribute)
        #print("subsets=",subsets)
        weighted_purities = []
        for v in self.attr_vals[attribute]:
            # don't calculate purity for v not in subset
            if v in subsets.groups.keys():
                #print("v=",v)
                s = subsets.get_group(v)
                #print("s=",s)
                weight = len(s) / len(S)
                #print("weight=",weight)
                purity = self.metric(s)
                #print("purity=",purity)
                weighted_purities.append(weight*purity)
        info_gain = purity_S - sum(weighted_purities)
        #print("info_gain=",info_gain)
        return info_gain, subsets

    def train(self, data, attr_vals):
        '''
        '''
        # must have at least one training instance
        assert len(data) > 0
        # find possible attribute vals from training data
        #self.attr_vals = get_attr_vals(data)
        self.attr_vals = attr_vals
        #print("self.attr_vals=",self.attr_vals)

        # attributes are number of cols except the label
        attributes = list(self.attr_vals.keys())[:-1]
        #print("attributes=",attributes)
        # attributes should be unique
        assert len(attributes) == len(set(attributes))
        #print("attributes=",attributes)
        #self.root = self.grow(data, attributes, depth=0)
        self.root = self.grow(data, attributes, depth=0)

    def predict_all(self, data):
        '''
        Make predictions for a list of test examples.
        '''
        preds = []
        for instance in data:
            #print("instance=",instance[1:])
            #preds += self.predict(self.root, instance[1:])
            # append a tuple of (prediction, weight)
            preds.append((self.predict(self.root, instance[1:]), instance[0]))
        return preds


    def predict(self, start, instance):
        '''
        Predict a label for a new instance by traversing the decision tree. 
        '''
        # if the node has children haven't reached a leaf yet.
        if start.children:
            #print("start.attribute=",start.attribute)
            #print('ENTERING LOOP')
            # find attr value of the test instance
            attr_val = instance[start.attribute]
            #print("attr_val=",attr_val)
            # which child to go to next
            branch = self.attr_vals[start.attribute].index(attr_val)
            #print("branch=",branch)
            nxt_node = start.children[branch]
            #print("nxt_node=",nxt_node)
            return self.predict(nxt_node, instance)
        return start.value

# ################# 
# Utility Functions
# #################
def split_data(S_df, attribute):
    '''
    Form a new subset of the data by splitting on an attribute.
    '''
    return S_df.groupby(attribute)


def majority_label(S):
    '''
    Return the majority label of a set S
    '''
    # group instances by label and get the mode
    # possibly more than 1 mode but we choose first by default.
    #label = S.iloc[:,S.columns[-1]].mode()[0]
    label = S.loc[:,S.columns[-1]].mode()[0]
    return label

