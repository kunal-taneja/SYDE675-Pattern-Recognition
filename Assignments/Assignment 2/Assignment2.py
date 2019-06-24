# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:53:52 2019

@author: Kunal Taneja
"""
#importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import random
from scipy.spatial import distance
import matplotlib.pyplot as plt
from collections import Counter
from keras.datasets import mnist


#extracting data from datasets
glass_data = pd.read_csv("glass.data",index_col = None, header = 0)
list1 = ["Id number","refractive index","Sodium","Magnesium","Aluminum","Silicon","Potassium","Calcium","Barium","Iron","Type"]
glass_data.columns = list1;
glass_data = glass_data.drop("Id number",axis =1)


tictac_data = pd.read_csv("tic-tac-toe.dat")
list2 = ["top-left-square","top-middle-square","top-right-square","middle-left-square","middle-middle-square","middle-right-square","bottom-left-square","bottom-middle-square","bottom-right-square","Class"]
tictac_data.columns = list2
le = LabelEncoder()
le.fit(tictac_data.Class.unique())
tictac_data.Class = le.transform(tictac_data.Class)

#Helper function to calculate mean value of all 784 features corresponding to each class, i.e we get 10x784 matrix of mean values.
def class_mean(X_train):
    mean_class = []
    for i in range(10):
        mean_class.append(np.mean(X_train[X_train['Label']==i]))
    return pd.DataFrame(mean_class)


#Helper function to determine the objective function for the bidirectional search
def interclass(y_f,f,X_train,mean_class,flag):
    temp = []

    if flag == 'SFS':
        temp.extend(y_f)
        temp.append(f)
        c_means = mean_class[mean_class.columns.intersection(temp)]
        global_mean = np.array(c_means.mean())
        global_mean = global_mean.reshape(1,global_mean.shape[0])
        dist = distance.cdist(c_means,global_mean,'euclidean')

    else:
        temp.extend(y_f)
        temp.remove(f)
        c_means = mean_class[mean_class.columns.intersection(temp)]
        global_mean = np.array(c_means.mean())
        global_mean = global_mean.reshape(1,global_mean.shape[0])
        dist = distance.cdist(c_means,global_mean,'euclidean')
    return dist.sum()


#implementation of bidirectional search for selecting desired number of features and then applying kNN over selected features to compare the accuracies.
def Q3a(X_train,X_test):
    y_f = []
    y_b=np.arange(784)
    all_f = np.arange(784)
    d = 392
    for1 = []
    back1 = []
    mean_class= class_mean(X_train)
    max_f = float('-Inf')
    min_b = float('-Inf')
    
    while (np.shape(y_f)[0] != d and np.shape(y_b)[0] != d):
        J = 0
        J1 = 0
        for index, f in np.ndenumerate(all_f):
            J = interclass(y_f,f,X_train,mean_class,'SFS')
            J1 = interclass(y_b,f,X_train,mean_class,'BFS')
            if(max_f<=J):
                best_feature = f
                max_f = J
            if(min_b<=J1):
                worst_feature = f
                min_b = J1
        for1.append(best_feature)
        back1.append(worst_feature)
        max_f = float('-Inf')
        min_b = float('-Inf')
        if (best_feature not in y_f) and (best_feature in y_b):  
            y_f = np.array(y_f)
            all_f = np.delete(all_f,np.where(all_f == best_feature))
            y_f = np.append(y_f,best_feature)
        if (worst_feature not in y_f and worst_feature in y_b):
            y_b = np.delete(y_b, np.where(y_b == worst_feature))
            all_f = np.delete(all_f,np.where(all_f == worst_feature))   
            
    important10 = np.zeros(784)
    important50 = np.zeros(784)
    important150 = np.zeros(784)
    important392 = np.zeros(784)
    unwanted = np.zeros(784)
    f=for1[0:10]
    f1 = for1[0:50]
    f2=for1[0:150]
    f3=for1[0:250]
    f4 = for1
    b=back1 
    features = [f,f1,f2,f3]
    accuracy_all = []
    for yy in range(len(features)):
        predictions = []
        data = X_train[X_train.columns.intersection(features[yy])]
        test_data = X_test[X_test.columns.intersection(features[yy])]
        distance1 = distance.cdist(test_data, data, 'euclidean')
        y = X_train.iloc[:,-1]
        test_y = X_test.iloc[:,-1]
        output = kNearestNeighbor(data.values, y.values, test_data.values, predictions,3,distance1)
        # transform the list into an array
        predictions = np.asarray(output)
        # evaluating accuracy
        accuracy = accuracy_score(test_y.values, predictions)
        accuracy_all.append(accuracy)
    list1= [accuracy_all[0], accuracy_all[1], accuracy_all[2], accuracy_all[3]]
    list2 = ['10','50','150','392']
    res=pd.DataFrame(list1)
    res.columns=["Accuracy"]
    res.index = list2
    print(res)

    
    for i in f:
        important10[i] = 1
    important10 = important10.reshape(28,28)
    for i in f1:
        important50[i] = 1
    important50 = important50.reshape(28,28)
    for i in f2:
        important150[i] = 1
    important150 = important150.reshape(28,28)

    for i in f4:
        important392[i] = 1
    important392 = important392.reshape(28,28)
    
    
    for i in b:
        unwanted[i] = 1
    unwanted= unwanted.reshape(28,28)
    
    
    
    fig, axs = plt.subplots(2, 3,figsize=(10,10))
    im = axs[0][0].imshow(important10, cmap = 'gray',interpolation='none')
    axs[0][0].set_title('First 10 Features')
    
    im = axs[0][1].imshow(important50, cmap = 'gray',interpolation='none')
    axs[0][1].set_title('First 50 Features')
    
    im = axs[0][2].imshow(important150,  cmap = 'gray',interpolation='none')
    axs[0][2].set_title('First 150 Features')
 
    im = axs[1][0].imshow(important392,  cmap = 'gray',interpolation='none')
    axs[1][1].set_title('First 392 Features')

    fig.colorbar(im,ax=axs.ravel().tolist())
    plt.show()


#Helper function to predict the results from kNN model.
def predict(X_train, y_train, x_test, k,distance1):
    # create list for distances and targets
    targets = []
    neighbor = np.argpartition(distance1, k)[:k]
    # make a list of the k neighbors' targets
    for i in neighbor:
        targets.append(y_train[i])

    # return most common target
    return Counter(targets).most_common(1)[0][0]

#Implementation of kNN classifier from scratch.
def kNearestNeighbor(X_train, y_train, X_test, predictions, k,distance1):

    # loop over all observations
    for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i, :], k,distance1[i,:]))
    return predictions


#Helper function to compute LDA from scratch.
def comp_mean_vectors(X, y):
    class_labels = np.unique(y)
    mean_vectors = []
    for cl in class_labels:
        mean_vectors.append(np.mean(X[y==cl], axis=0))
    return mean_vectors


#Helper function to evaluate S_w metric of LDA 
def scatter_within(X, y):
    class_labels = np.unique(y)
    n_features = X.shape[1]
    mean_vectors = comp_mean_vectors(X.values, y)
    S_W = np.zeros((n_features, n_features))
    for cl, mv in zip(class_labels, mean_vectors):
        class_sc_mat = np.zeros((n_features, n_features))                 
        for row in X[y == cl].values:
            row, mv = row.reshape(n_features, 1), mv.reshape(n_features, 1)
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat                           
    return S_W

#Helper function to evaluate S_b metric of LDA 
def scatter_between(X, y):
    overall_mean = np.mean(X.values, axis=0)
    n_features = X.shape[1]
    mean_vectors = comp_mean_vectors(X.values, y)    
    S_B = np.zeros((n_features, n_features))
    for i, mean_vec in enumerate(mean_vectors):  
        n = X[y==i+1].shape[0]
        mean_vec = mean_vec.reshape(n_features, 1)
        overall_mean = overall_mean.reshape(n_features, 1)
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    return S_B


#Helper function to get transformed dimensions
def get_components(X,eig_vals, eig_vecs, n_comp):
    n_features = X.shape[1]
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    W = np.hstack([eig_pairs[i][1].reshape(n_features, 1) for i in range(0, n_comp)])
    return W
     
"""defining helper functions used to construct C4.5 Tree"""
#Helper function to get threshold values for each feature.
def get_threshold_values(data):
    
    thresholds = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):
        thresholds[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)

        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_threshold = (current_value + previous_value) / 2
                
                thresholds[column_index].append(potential_threshold)
        if (unique_values.shape[0] == 1):
            potential_threshold = unique_values[index]
            
            thresholds[column_index].append(potential_threshold)

    
    return thresholds

#Split the data based on the feature and the threshold value as data above the threshold and data below the threshold value.
def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values >  split_value]
    
    return data_below, data_above


def data_split_cat(data,i):
  data_x,data_o,data_b = [],[],[]
  data_x = data[data.iloc[:,i]=='x']
  data_o = data[data.iloc[:,i]=='o']
  data_b = data[data.iloc[:,i]=='b']
  
  return data_x,data_o,data_b





#helper function to calculate the entropy for feature passed as an argument
def get_entropy(data): 
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy

def get_entropy_tic(data,label):
    
    label_column = label
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy

#helper function to calculate the information gain from calculated entropies
def information_gain(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy =  (p_data_below * get_entropy(data_below) 
                      + p_data_above * get_entropy(data_above))
    
    return overall_entropy


def calculate_overall_entropy_cat(data_o, data_x,data_b):
    
    n = len(data_o) + len(data_x) + len(data_b)
    p_attr_x = len(data_x) / n
    p_attr_o = len(data_o) / n
    p_attr_b = len(data_b) / n
 
    overall_entropy =  (p_attr_x * get_entropy_tic(p_attr_x,data_x.values[0:,-1])) + p_attr_o * get_entropy_tic(p_attr_o,data_o.values[0:,-1])+ (p_attr_b * get_entropy_tic(p_attr_o,data_b.values[0:,-1]))
     
    
    return overall_entropy


#Check if all data is of same class and we can classify this subset of data finally.
def check_purity(data):
    
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False

#Classify data based on majority voting only when no rule is left to classify the data
#Hence we classify remaining datapoints based on the majority voting.
def majority_voting(data):
    
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification

#Gives the best feature and its split value after checking all features based on gain ratio
def best_splitting_feature(data, thresholds):
    
    entropy_label = get_entropy(data)   
    overall_gain = -1.0
    for column_index in thresholds:
        for value in thresholds[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = information_gain(data_below, data_above)
            current_information_gain = entropy_label - current_overall_entropy
            current_splitting_info = get_splitting_information(data_below,data_above)
            if current_splitting_info == 0:
                current_gain_ratio = 0
            else:
                current_gain_ratio = float(current_information_gain / current_splitting_info)

            if current_gain_ratio >= overall_gain:
                overall_gain = current_gain_ratio
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value

#Calculates the splitting Info of data above and below for that threshold value
def get_splitting_information(data_below,data_above):
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below)/ n
    p_data_above = len(data_above) / n

    if p_data_below == 0:
        splitting_info = p_data_above * np.log2(p_data_above)
    elif p_data_above == 0:
        splitting_info = p_data_below * np.log2(p_data_below)
    else:
        splitting_info = -p_data_below * np.log2(p_data_below) -p_data_above * np.log2(p_data_above) 
    
    return splitting_info

#helper function defining structure of the tree
parent_node = None
class NodeStructureGlass():
    def __init__(self, is_leaf, classification, attribute_threshold, parent, left_child, right_child, height):

        self.parent = parent
        self.left_child = None
        self.right_child = None
        self.is_leaf = True
        self.classification = None
        self.attribute_split = None
        self.attribute_threshold = None
        self.height = None
        
        
class NodeStructureTicTac():
    def __init__(self, is_leaf_node, classification, parent, b_child, o_child,x_child, height):

        self.classification = None
        self.attribute_split = None
        self.parent = parent
        self.b_child = None
        self.o_child = None
        self.x_child = None
        self.height = None
        self.is_leaf_node = True
        

#helper function which defines C45 tree structure
def C45_tree(df, parent_node,counter=0, min_samples=3):
    node = NodeStructureGlass(True, None, None, parent_node, None, None, 0) #is_leaf is set ot be true when creating a new node.
    if counter == 0:
        global FEATURES
        FEATURES = df.columns
        data = df.values
    else:
        data = df           
    
    
    # base case to exit recursion stack which executes when node is a leaf or thier are no rules left to classify the data. 
    if (check_purity(data)) or (len(data) < min_samples):
        classification = majority_voting(data)
        node.is_leaf = True
        node.classification = classification
        return node

    
    # recursive part to iterate over all the paths of the tree.
    else:    
        counter += 1

        # helper functions 
        thresholds = get_threshold_values(data)
        split_column, split_value = best_splitting_feature(data, thresholds)
        data_below, data_above = split_data(data, split_column, split_value)
        node.is_leaf = False
        # instantiate sub-tree
        feature_name = FEATURES[split_column]
        
        
        if (parent_node == None):
            node.height = 0
        else:
            node.parent = parent_node
            node.height = node.parent.height + 1


        node.attribute_split = feature_name
        node.attribute_threshold = split_value

        # find answers (recursion)
        node.left_child = C45_tree(data_below,node, counter, min_samples)
        node.right_child = C45_tree(data_above,node, counter, min_samples)
        
        return node
    
    
    
def C45_tree_tic(data,parent_node):
    node = NodeStructureTicTac(True, None, None, parent_node, None, None, 0)
    #Base-Case
    if data.empty:
        return
    if (data.shape[1] == 1):
        counts = np.bincount(data.iloc[:,-1])
        node.is_leaf_node = True
        node.classification = np.argmax(counts)
        # node.height = node.parent.height + 1
        return node
    elif len(data.iloc[:,-1].unique()) == 1:
        node.is_leaf_node = True
        node.classification = data.iloc[:,-1].unique()[0]
        return node

    labels = data.iloc[:,-1].values
    datapoint = data.iloc[:,0:-1].values
    entropy_dec = get_entropy_tic(datapoint,labels)
    max_gainratio = float('-Inf')
    for i in range(data.keys().size-1):
        data_x,data_o,data_b = data_split_cat(data,i)
        p_attr_x = len(data_x) / data.shape[0]
        p_attr_o = len(data_o) / data.shape[0]
        p_attr_b = len(data_b) / data.shape[0]

        infogain = entropy_dec - calculate_overall_entropy_cat(data_o, data_x,data_b)

        splitinfo = -(p_attr_x * np.ma.log2(p_attr_x))-(p_attr_o * np.ma.log2(p_attr_o))-(p_attr_b * np.ma.log2(p_attr_b))
        if np.ma.is_masked(splitinfo):
            gainratio = 0
        else:
            gainratio = infogain / splitinfo 
        if (gainratio >= max_gainratio):
            max_gainratio = gainratio
            fname = data.keys()[i]
        node.is_leaf_node = False
        if (parent_node == None):
            node.height = 0
        else:
            node.parent = parent_node
            node.height = node.parent.height + 1

    node.attribute_split = fname

    dx = data[data[fname] == 'x']
    dx = dx.drop(columns=fname)
    do = data[data[fname] == 'o']
    do = do.drop(columns=fname)
    db = data[data[fname] == 'b']
    db = db.drop(columns=fname)
    node.b_child = C45_tree_tic(dx,node)
    node.o_child = C45_tree_tic(do,node)
    node.x_child = C45_tree_tic(db,node)


    return node

    
#helper function to convert tree structure into equivalent if and else rules.
def tree_to_rules(root, path, pathlen,all_paths,val):
    if (root==None): #empty tree
        return
    
    if root.is_leaf == True: #end of rule
        path.append(root.classification) 
    else:
        path.append('row[\'' + root.attribute_split + '\']' + val + str(root.attribute_threshold)) #appending conditions of the rule
        
    pathlen= pathlen+1
    if (root.left_child == None and root.right_child == None): # If leaf, append current path
        add = path[:]
        all_paths.append(add)
        path.pop()
        root = root.parent
    else:
        tree_to_rules(root.left_child, path, pathlen,all_paths,' <= ')
        path[pathlen-1]= 'row[\'' + root.attribute_split + '\']' +' > ' + str(root.attribute_threshold)
        tree_to_rules(root.right_child, path,pathlen,all_paths,' <= ')
        path.pop()

    return all_paths


def tree_to_rules_tic(root, path, pathlen,all_paths):
    if (root==None):
        return
    
    if root.is_leaf_node == True: 
        path.append(root.classification) 
    else:
        path.append((root.attribute_split))
        
    pathlen= pathlen+1
    if (root.b_child == None and root.o_child == None and root.x_child == None): # If leaf, append current path
        add = path[:]
        all_paths.append(add)
        path.pop()
        root = root.parent
    else:
        path.append('b')
        tree_to_rules_tic(root.b_child, path, pathlen,all_paths)
        path.pop()
        path.append('o')
        tree_to_rules_tic(root.o_child, path,pathlen,all_paths)
        path.pop()
        path.append('x')
        tree_to_rules_tic(root.x_child, path,pathlen,all_paths)
        path.pop()
        path.pop()

    return all_paths

#helper function to calculate the accuracy of each selected rule against the validation data.
def calc_accuracy_rule(rule,test):
    wrong = 0
#Checking correctly classified samples.
    for index, row in test.iterrows():
        s=0
        while(s<len(rule)-1):
            if (eval(rule[s])== False):
                wrong += 1
                break
            s=s+1
    #Initial Accuracy of one Rule before pruning
    accuracy = (test.shape[0]-wrong) / test.shape[0]
    return accuracy

def calc_accuracy_rule_tic(rule,test):
    wrong = 0
#Checking correctly classified samples.
    for index, row in test.iterrows():
        s=0
        while(s<len(rule)-2):
            if (row[rule[s]] != rule[s+1]):
                wrong += 1
                break
            s=s+2
    #Initial Accuracy of one Rule before pruning
    accuracy = (test.shape[0]-wrong) / test.shape[0]
    return accuracy

#Helper function which prune_trees the C4.5 rules    
def prune_tree(all_rules,val_data):
    acc_rlist = []
    rule_accuracies = []
    rule_index = []
    #What are the labels in my val data
    ctoprune_tree = val_data['Type'].unique()
#     Loop at all rules one by one
    for i in range(len(all_rules)):
        init_accuracy = 0
        #Loop only on the rules applicable to my valset
        if all_rules[i][-1] in ctoprune_tree:
                #Get the label of the Rule
                label = all_rules[i][-1]
                #Get all samples for that label
                test = val_data[val_data['Type']==label]
                #Check Initial Accuracy of the rule
                init_accuracy = calc_accuracy_rule(all_rules[i],test)
                
                temp = all_rules[i][:]
                prune_treed_accuracy = -1
                while (init_accuracy!=prune_treed_accuracy):
                    for x in range(len(all_rules[i])-1):
                        del temp[x]
                        accuracy = calc_accuracy_rule(temp,test)
                        if accuracy > init_accuracy:
                            delx = x
                            init_accuracy = accuracy
                        temp = all_rules[i][:]
                    # Ensure variable is defined
                    try:
                        delx
                    except NameError:
                        delx = None

                    if delx is not None:
                        del all_rules[i][delx]
                        del delx
                        # prune_treed_accuracy = init_accuracy
                        if (len(all_rules[i])== 2):
                            prune_treed_accuracy = init_accuracy
                    else:
                        prune_treed_accuracy = init_accuracy
        else:
            prune_treed_accuracy = init_accuracy
        acc_rlist.append(prune_treed_accuracy)
        rule_index.append(i)
    rule_accuracies.append(acc_rlist)
    rule_accuracies.append(rule_index)
    rule_accuracies= np.array(rule_accuracies)
    rule_accuracies = pd.DataFrame(rule_accuracies.T)
    rule_accuracies = rule_accuracies.sort_values(0,ascending=False)
    rule_accuracies = pd.DataFrame(rule_accuracies)
    return all_rules,rule_accuracies


def prune_true_tic(all_rules,val_data):
    acc_rlist = []
    rule_accuracies = []
    rule_index = []
    #What are the labels in my val data
    ctoprune = val_data['Class'].unique()

    #Loop at all rules one by one
    for i in range(len(all_rules)):
        init_accuracy = 0
        #Loop only on the rules applicable to my valset
        if all_rules[i][-1] in ctoprune:
                #Get the label of the Rule
                label = all_rules[i][-1]
                #Get all samples for that label
                test = val_data[val_data['Class']==label]
                #Check Initial Accuracy of the rule  
                temp = all_rules[i][:]
                pruned_accuracy = -1
                while (init_accuracy!=pruned_accuracy):
                    for x in range(0,len(all_rules[i])-2,2):
                        del temp[x]
                        del temp[x]
                        accuracy = calc_accuracy_rule_tic(temp,test)
                        if accuracy > init_accuracy:
                            delx = x
                            init_accuracy = accuracy
                        temp = all_rules[i][:]
                    # Ensure variable is defined
                    try:
                        delx
                    except NameError:
                        delx = None

                    if delx is not None:
                        del all_rules[i][delx]
                        del all_rules[i][delx]
                        del delx
                        if (len(all_rules[i])== 3):
                            pruned_accuracy = init_accuracy
                    else:
                        pruned_accuracy = init_accuracy
        else:
            pruned_accuracy = init_accuracy
        acc_rlist.append(pruned_accuracy)
        rule_index.append(i)
    rule_accuracies.append(acc_rlist)
    rule_accuracies.append(rule_index)
    rule_accuracies= np.array(rule_accuracies)
    rule_accuracies = pd.DataFrame(rule_accuracies.T)
    rule_accuracies = rule_accuracies.sort_values(0,ascending=False)
    rule_accuracies = pd.DataFrame(rule_accuracies)
    return all_rules,rule_accuracies


#Helper function to check the accuracy of C4.5 tree obtained after pruning.                
def predict_pruned(all_rules,test_data,rule_accuracies):
    answer = []
    unclassified = 0
    for index, row in test_data.iterrows():
        for indo,valus in rule_accuracies.iterrows():
            rule = all_rules[int(valus[1])]
            s=0
            count = 0
            while(s<len(rule)-1):
                if (eval(rule[s])== True):
                    count = count + 1
                s = s+1
            if (count == len(rule)-1):
                prediction = rule[-1]
                break
        try:
            prediction
        except NameError:
            prediction = None
        if prediction is not None:
            answer.append(prediction)
            del prediction
        else:
            unclassified = unclassified + 1
    return answer


def predict_prunedtic(all_rules,test_data,maping):
    answer = []
    unclassified = 0
    for index, row in test_data.iterrows():
        # prediction = row[-1]
        for indo,valus in maping.iterrows():
            rule = all_rules[int(valus[1])]
            s=0
            count = 0
            while(s<len(rule)-2):
                if (row[rule[s]] == rule[s+1]):
                    count = count + 1
                s = s+2
            if (count == (len(rule)-1)/2):
                prediction = rule[-1]
                break
        try:
            prediction
        except NameError:
            prediction = None
        if prediction is not None:
            answer.append(prediction)
            del prediction
        else:
            unclassified = unclassified + 1
    # print('No of Unclassified:',unclassified)
    return answer



#Helper function to check the accuracy of C4.5 tree obtained before pruning.
def predict_prepruned(all_rules,test_data):
    answer = []
    unclassified = 0
    for index, row in test_data.iterrows():
        # prediction = row[-1]
        for rule in all_rules:
            s=0
            count = 0
            while(s<len(rule)-1):
                if (eval(rule[s])== True):
                    count = count + 1
                s = s+1
            if (count == len(rule)-1):
                prediction = rule[-1]
                break
        try:
            prediction
        except NameError:
            prediction = None
        if prediction is not None:
            answer.append(prediction)
            del prediction
        else:
            unclassified = unclassified + 1
    return answer


def predict_preprunedtic(all_rules,test_data):
    answer = []
    unclassified = 0
    for index, row in test_data.iterrows():
        # prediction = row[-1]
        for rule in all_rules:
            s=0
            count = 0
            while(s<len(rule)-2):
                if (row[rule[s]] == rule[s+1]):
                    count = count + 1
                s = s+2
            if (count == (len(rule)-1)/2):
                prediction = rule[-1]
                break
        try:
            prediction
        except NameError:
            prediction = None
        if prediction is not None:
            answer.append(prediction)
        else:
            unclassified = unclassified + 1
    # print('No of Unclassified:',unclassified)
    return answer

#helper function to make final predictions over the test data after pruning
def predicting_test(root,data):
    predictions = []
    tree = root 
    data = data.iloc[:, :-1]
    for index, sample in data.iterrows():
        root = tree
        while(tree.is_leaf!=True):
            if (sample.loc[tree.attribute_split] <= tree.attribute_threshold):
                tree = tree.left_child
            else:
                tree = tree.right_child
        predictions.append(tree.classification)
        tree = root

    return predictions

def predicting_test_tic(root,data):
    predictions = []
    tree = root 
    data = data.iloc[:, :-1]
    for index, sample in data.iterrows():
        root = tree
        while(tree.is_leaf_node!=True):
            if (sample.loc[tree.attribute_split] == 'b'):
                tree = tree.b_child
            elif(sample.loc[tree.attribute_split] == 'o'):
                tree = tree.o_child
            else:
                tree = tree.x_child
        predictions.append(tree.classification)
        tree = root

    return predictions


def misclassified_noise(num,data):

    siz_d = data.shape[0]
    indx = int((num * siz_d)/100)
    for x in range(indx):
        pick = random.randint(0,int(siz_d/2))
        label = data.iloc[pick:,-1].values[0]
        if label > 0:
            data.iloc[pick:,-1] = data.iloc[pick:,-1] - 1
        else:
            data.iloc[pick:,-1] = data.iloc[pick:,-1] + 1
    return data

def contradictory_noise(num,data):
    siz_d = data.shape[0]
    indx = int((num * siz_d)/100)
    count = 0
    for x in range(indx):
        count = count+1
        pick = random.randint(0,siz_d)
        label = data.iloc[pick:,-1].values[0]
        if label > 0:
            label = label + 1
        else:
            label = label - 1
        last = data.shape[0]  
        data = data.append(data.iloc[pick,:])
        data.iloc[last,-1] = label
    return data




def noise_glass(train_data,test_data,val_data):
        noise1_5 = []
        noise1_10 = []
        noise1_15 = []
        noise2_5 = []
        noise2_10 = []
        noise2_15 = []
        #Noisy Data
        train_data = contradictory_noise(5,train_data)
                #Form the tree
        tree = C45_tree(train_data,parent_node)
        #Get all the rules
        all_rules = tree_to_rules(tree,[],0,[],' <= ')
        y_true= test_data.iloc[:,-1].values
        answer = predict_prepruned(all_rules,test_data)
        accuracy = accuracy_score(y_true, answer)
        all_rules,maping = prune_tree(all_rules,val_data)
        answer = predict_pruned(all_rules,test_data,maping)
        y_true= test_data.iloc[:,-1].values
        accuracy = accuracy_score(y_true, answer)
        noise1_5.append(accuracy)
    
    
        train_data = contradictory_noise(10,train_data)
        tree = C45_tree(train_data,parent_node)
        #Get all the rules
        all_rules = tree_to_rules(tree,[],0,[],' <= ')
        y_true= test_data.iloc[:,-1].values
        answer = predict_prepruned(all_rules,test_data)
        accuracy = accuracy_score(y_true, answer)
        all_rules,maping = prune_tree(all_rules,val_data)
        answer = predict_pruned(all_rules,test_data,maping)
        y_true= test_data.iloc[:,-1].values
        accuracy = accuracy_score(y_true, answer)
        noise1_10.append(accuracy)
    
    
        train_data = contradictory_noise(15,train_data)
        tree = C45_tree(train_data,parent_node)
        #Get all the rules
        all_rules = tree_to_rules(tree,[],0,[],' <= ')
        y_true= test_data.iloc[:,-1].values
        answer = predict_prepruned(all_rules,test_data)
        accuracy = accuracy_score(y_true, answer)
        all_rules,maping = prune_tree(all_rules,val_data)
        answer = predict_pruned(all_rules,test_data,maping)
        y_true= test_data.iloc[:,-1].values
        accuracy = accuracy_score(y_true, answer)
        noise1_15.append(accuracy)
    
    
    
        #Noisy Data2
        train_data = misclassified_noise(5,train_data)
        tree = C45_tree(train_data,parent_node)
        #Get all the rules
        all_rules = tree_to_rules(tree,[],0,[])
        y_true= test_data.iloc[:,-1].values
        answer = predict_prepruned(all_rules,test_data)
        accuracy = accuracy_score(y_true, answer)
        all_rules,maping = prune_tree(all_rules,val_data)
        answer = predict_pruned(all_rules,test_data,maping)
        y_true= test_data.iloc[:,-1].values
        accuracy = accuracy_score(y_true, answer)
        noise2_5.append(accuracy)
    
    
    
        train_data = misclassified_noise(10,train_data)
        tree = C45_tree(train_data,parent_node)
        #Get all the rules
        all_rules = tree_to_rules(tree,[],0,[])
        y_true= test_data.iloc[:,-1].values
        answer = predict_prepruned(all_rules,test_data)
        accuracy = accuracy_score(y_true, answer)
        all_rules,maping = prune_tree(all_rules,val_data)
        answer = predict_pruned(all_rules,test_data,maping)
        y_true= test_data.iloc[:,-1].values
        accuracy = accuracy_score(y_true, answer)
        noise2_10.append(accuracy)
    
    
        train_data = misclassified_noise(15,train_data)
        tree = C45_tree(train_data,parent_node)
        #Get all the rules
        all_rules = tree_to_rules(tree,[],0,[])
        y_true= test_data.iloc[:,-1].values
        answer = predict_prepruned(all_rules,test_data)
        accuracy = accuracy_score(y_true, answer)
        all_rules,maping = prune_tree(all_rules,val_data)
        answer = predict_pruned(all_rules,test_data,maping)
        y_true= test_data.iloc[:,-1].values
        accuracy = accuracy_score(y_true, answer)
        noise2_15.append(accuracy)
        
        
def noise_tic(train_data,test_data,val_data):
        noise1_5 = []
        noise1_10 = []
        noise1_15 = []
        noise2_5 = []
        noise2_10 = []
        noise2_15 = []
        #Noisy Data
        train_data = contradictory_noise(5,train_data)
                #Form the tree
        tree = C45_tree_tic(train_data,parent_node)
        #Get all the rules
        all_rules = tree_to_rules_tic(tree,[],0,[])
        y_true= test_data.iloc[:,-1].values
        answer = predict_preprunedtic(all_rules,test_data)
        accuracy = accuracy_score(y_true, answer)
        all_rules,maping = prune_true_tic(all_rules,val_data)
        answer = predict_prunedtic(all_rules,test_data,maping)
        y_true= test_data.iloc[:,-1].values
        accuracy = accuracy_score(y_true, answer)
        noise1_5.append(accuracy)
    
    
        train_data = contradictory_noise(10,train_data)
        tree = C45_tree_tic(train_data,parent_node)
        #Get all the rules
        all_rules = tree_to_rules_tic(tree,[],0,[])
        y_true= test_data.iloc[:,-1].values
        answer = predict_preprunedtic(all_rules,test_data)
        accuracy = accuracy_score(y_true, answer)
        all_rules,maping = prune_true_tic(all_rules,val_data)
        answer = predict_prunedtic(all_rules,test_data,maping)
        y_true= test_data.iloc[:,-1].values
        accuracy = accuracy_score(y_true, answer)
        noise1_10.append(accuracy)
    
    
        train_data = contradictory_noise(15,train_data)
        tree = C45_tree_tic(train_data,parent_node)
        #Get all the rules
        all_rules = tree_to_rules_tic(tree,[],0,[])
        y_true= test_data.iloc[:,-1].values
        answer = predict_preprunedtic(all_rules,test_data)
        accuracy = accuracy_score(y_true, answer)
        all_rules,maping = prune_true_tic(all_rules,val_data)
        answer = predict_prunedtic(all_rules,test_data,maping)
        y_true= test_data.iloc[:,-1].values
        accuracy = accuracy_score(y_true, answer)
        noise1_15.append(accuracy)
    
    
    
        #Noisy Data2
        train_data = misclassified_noise(5,train_data)
        tree = C45_tree_tic(train_data,parent_node)
        #Get all the rules
        all_rules = tree_to_rules_tic(tree,[],0,[])
        y_true= test_data.iloc[:,-1].values
        answer = predict_preprunedtic(all_rules,test_data)
        accuracy = accuracy_score(y_true, answer)
        all_rules,maping = prune_true_tic(all_rules,val_data)
        answer = predict_prunedtic(all_rules,test_data,maping)
        y_true= test_data.iloc[:,-1].values
        accuracy = accuracy_score(y_true, answer)
        noise2_5.append(accuracy)
    
    
    
        train_data = misclassified_noise(10,train_data)
        tree = C45_tree_tic(train_data,parent_node)
        #Get all the rules
        all_rules = tree_to_rules_tic(tree,[],0,[])
        y_true= test_data.iloc[:,-1].values
        answer = predict_preprunedtic(all_rules,test_data)
        accuracy = accuracy_score(y_true, answer)
        all_rules,maping = prune_true_tic(all_rules,val_data)
        answer = predict_prunedtic(all_rules,test_data,maping)
        y_true= test_data.iloc[:,-1].values
        accuracy = accuracy_score(y_true, answer)
        noise2_10.append(accuracy)
    
    
        train_data = misclassified_noise(15,train_data)
        tree = C45_tree_tic(train_data,parent_node)
        #Get all the rules
        all_rules = tree_to_rules_tic(tree,[],0,[])
        y_true= test_data.iloc[:,-1].values
        answer = predict_preprunedtic(all_rules,test_data)
        accuracy = accuracy_score(y_true, answer)
        all_rules,maping = prune_true_tic(all_rules,val_data)
        answer = predict_prunedtic(all_rules,test_data,maping)
        y_true= test_data.iloc[:,-1].values
        accuracy = accuracy_score(y_true, answer)
        noise2_15.append(accuracy)
        
        
def Q1_glass():
    meanacc= []
    rule_accuracies = []
    acc_pruning = []
    #10 times 10 fold for better results
    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
    for train_index, test_index in rkf.split(glass_data):
        parent_node = None
        test_data = glass_data.iloc[test_index]
        train_data,val_data= train_test_split(glass_data.iloc[train_index], test_size=0.2)
        
        tree = C45_tree(train_data,parent_node)
        all_rules = tree_to_rules(tree,[],0,[],' <= ')

        accuracy = []
        #Pre-Pruning Accuracy of Test Data With Rules
        y_true= test_data.iloc[:,-1].values
        answer = predict_prepruned(all_rules,test_data)
        accuracy = accuracy_score(y_true, answer)
        acc_pruning.append(accuracy)
        
        
        #Post Pruning Accuracy with Rules
        all_rules,rule_accuracies = prune_tree(all_rules,val_data)
        answer = predict_pruned(all_rules,test_data,rule_accuracies)
        accuracy = accuracy_score(y_true, answer)
        meanacc.append(accuracy)
        
           
    print('The Mean Accuracy of C45 for glass_data',sum(acc_pruning) / len(acc_pruning))
    print('The Mean Variance of C45 for glass_data',np.var(np.array(acc_pruning)))
    print('Done with Glass Data')
    
def Q1_tic():
    meanacc = []
    acc_pruning = []
    #10 times 10 fold for better results
    rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
    for train_index, test_index in rkf.split(tictac_data):
        test_data = tictac_data.iloc[test_index]
        train_data,val_data= train_test_split(tictac_data.iloc[train_index], test_size=0.2)
        parent_node = None
        #Form the tree
        tree = C45_tree_tic(train_data,parent_node)
        #Get all the rules
        all_rules = tree_to_rules_tic(tree,[],0,[])
        y_true= test_data.iloc[:,-1].values
        answer = predict_preprunedtic(all_rules,test_data)
        accuracy = accuracy_score(y_true, answer)
        acc_pruning.append(accuracy)

        #Post Pruning Accuracy with Rules
        all_rules,maping = prune_true_tic(all_rules,val_data)
        answer = predict_prunedtic(all_rules,test_data,maping)
        y_true= test_data.iloc[:,-1].values
        accuracy = accuracy_score(y_true, answer)
        meanacc.append(accuracy)
        
     
    print('The Mean Accuracy of C45 for tic_tac_toe data',sum(meanacc) / len(meanacc))
    print('The Mean Variance of C45 for tic_tac_toe data',np.var(np.array(meanacc)))
    print('Done with Question1 and 2')
    
def main():
    Q1_glass()
    Q1_tic()
    print('Starting with question 3')
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = pd.DataFrame(X_train.reshape(60000,784))
    X_test = pd.DataFrame(X_test.reshape(10000,784))
    X_train['Label'] = y_train.reshape(60000,1)
    X_test['Label'] = y_test.reshape(10000,1)
    X_train = X_train.head(10000)
    y_train = y_train[:10000]
    S_W, S_B = scatter_within(X_train.iloc[:,0:784], y_train), scatter_between(X_train.iloc[:,0:784], y_train)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
    
    Q3a(X_train,X_test)
    
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        
    lda_accuracies = list()
    for xx in range(9):
        predictions = []
        W = get_components(X_train.iloc[:,0:784].values,eig_vals, eig_vecs, xx+1)
        X_lda = X_train.iloc[:,0:784].values.dot(W)
        #print('Shape of train for components',xx,'is',X_lda.shape)
        test_lda = X_test.iloc[:,0:784].values.dot(W)
        #print('Shape of test for components',xx,'is',test_lda.shape)
        distance1 = distance.cdist(test_lda, X_lda, 'euclidean')
        y_train = X_train.iloc[:,-1]
        test_y = X_test.iloc[:,-1]
        output = kNearestNeighbor(X_lda, y_train.values, test_lda, predictions,3,distance1)
        # transform the list into an array
        predictions = np.asarray(output)
        # evaluating accuracy
        accuracy = accuracy_score(test_y.values, predictions)
        lda_accuracies.append(accuracy)
    list1= [lda_accuracies[0], lda_accuracies[1], lda_accuracies[2], lda_accuracies[3], lda_accuracies[4], lda_accuracies[5], lda_accuracies[6], lda_accuracies[7], lda_accuracies[8]]
    list2 = ['LDA_1','LDA_2','LDA_3','LDA_4','LDA_5','LDA_6','LDA_7','LDA_8','LDA_9']
    res=pd.DataFrame(list1)
    res.columns=["Accuracy"]
    res.index = list2
    print(res)
    print('done with this assignment')

if __name__ == "__main__":
    main()


