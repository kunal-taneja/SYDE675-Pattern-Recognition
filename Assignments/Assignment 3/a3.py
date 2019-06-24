# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:57:03 2019

@author: Kunal Taneja
"""
#importing basic python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing required libraries from sklearn
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

#importing required datasets
dataA_q2 = pd.read_csv('classA.csv',header=None)
dataB_q2= pd.read_csv('classB.csv',header=None)
dataA_q1 = pd.read_csv('q1_classA.csv',header=None)
dataB_q1 = pd.read_csv('q1_classB.csv',header=None)

#preparing data by appending suitable labels along with the features.
dataA_q1['Label'] = np.ones(np.shape(dataA_q1[0]))
dataB_q1['Label'] = np.zeros(np.shape(dataB_q1[0]))
dataA_q2['Label'] = np.ones(np.shape(dataA_q2[0]))
dataB_q2['Label'] = np.zeros(np.shape(dataB_q2[0]))
all_data = [dataA_q1,dataB_q1]
q1data = pd.concat(all_data,ignore_index=True)
all_data2= [dataA_q2,dataB_q2]
q2_data = pd.concat(all_data2,ignore_index=True)
    
#functions to plot decision boundaries for the SVM classifications
def boundary_mesh(x, y):

    x_min= x.min() - 1
    x_max = x.max() + 1
    y_min = y.min() - 1
    y_max = y.max() + 1
    # step size of 0.5 taken in the meshgrid
    step = 0.5
    
    #making  a grid of size 10x10 using np.arrange function in numpy
    X, Y = np.meshgrid(np.arange(x_min, x_max, step),np.arange(y_min, y_max, step))
    return X,Y

# function to plot decision boundary for SVM classification
def decision_boundary(svm_model, x_min, x_max):
    support_vector1 = svm_model.coef_[0]
    support_vector2 = svm_model.intercept_[0]
    
    x_o = np.linspace(x_min, x_max, 200)
    db = -support_vector1[0]/support_vector1[1] * x_o - support_vector2/support_vector1[1]

    margin = 1/support_vector1[1]
    gutter_1 = db + margin
    gutter_2 = db - margin

    svm_supports = svm_model.support_vectors_
    plt.scatter(svm_supports[:, 0], svm_supports[:, 1], s=100, facecolors='#FAAAAA')
    plt.plot(x_o, db, "k-", linewidth=2)
    plt.plot(x_o, gutter_1, "k--", linewidth=2)
    plt.plot(x_o, gutter_2, "k--", linewidth=2)
  

def q1a():
    
    plt.figure(figsize=(10,5))
    plt.scatter(dataA_q1.iloc[:,0],dataA_q1.iloc[:,1], c='g', marker="+", label='dataA_q1')
    plt.scatter(dataB_q1.iloc[:,0],dataB_q1.iloc[:,1], c='b', marker="_", label='dataB_q1')
    plt.legend(loc='lower left')
    plt.show()

def q1b_c():
    
    X = q1data.iloc[:, :2].values
    y = q1data.iloc[:, -1].values
    
    svm_model_1 = svm.SVC(kernel='linear',C=0.001).fit(X, y) 
    svm_model_2 = svm.SVC(kernel='linear',C=0.01).fit(X, y) 
    svm_model_3 =  svm.SVC(kernel='linear',C=0.1).fit(X, y) 
    svm_model_4 = svm.SVC(kernel='linear',C=1).fit(X, y) 
    
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    decision_boundary(svm_model_1, X.min(), X.max())
    plt.scatter(dataA_q1.iloc[:,0],dataA_q1.iloc[:,1], c='g', marker="+", label='dataA_q1')
    plt.scatter(dataB_q1.iloc[:,0],dataB_q1.iloc[:,1], c='b', marker="_", label='dataB_q1')
    plt.title('Linear SVM - C:0.001')
    plt.axis([240, 402, 50, 320])
    plt.legend(loc='lower left')
    plt.show()
    
    
    plt.subplot(122)
    plt.clf()
    decision_boundary(svm_model_2, X.min(), X.max())
    plt.scatter(dataA_q1.iloc[:,0],dataA_q1.iloc[:,1], c='g', marker="+", label='dataA_q1')
    plt.scatter(dataB_q1.iloc[:,0],dataB_q1.iloc[:,1], c='b', marker="_", label='dataB_q1')
    plt.title('Linear SVM - C:0.01')
    plt.legend(loc='lower left')
    plt.axis([240, 402, 50, 320])
    plt.show()
    
    
    plt.subplot(221)
    plt.clf()
    decision_boundary(svm_model_3, X.min(), X.max())
    plt.scatter(dataA_q1.iloc[:,0],dataA_q1.iloc[:,1], c='g', marker="+", label='dataA_q1')
    plt.scatter(dataB_q1.iloc[:,0],dataB_q1.iloc[:,1], c='b', marker="_", label='dataB_q1')
    plt.title('Linear SVM - C:0.1')
    plt.axis([240, 402, 50, 320])
    plt.legend(loc='lower left')
    plt.show()
   
    
    plt.subplot(222)
    plt.clf()
    decision_boundary(svm_model_4, X.min(), X.max())
    plt.scatter(dataA_q1.iloc[:,0],dataA_q1.iloc[:,1], c='g', marker="+", label='dataA_q1')
    plt.scatter(dataB_q1.iloc[:,0],dataB_q1.iloc[:,1], c='b', marker="_", label='dataB_q1')
    plt.title('Linear SVM - C:1')
    plt.axis([240, 402, 50, 320])
    plt.legend(loc='lower left')
    plt.show()
    plt.clf()

def q2a():
    
    plt.figure(figsize=(10,5))
    plt.scatter(dataA_q2.iloc[:,0],dataA_q2.iloc[:,1], c='g', marker="+", label='dataA_q2')
    plt.scatter(dataB_q2.iloc[:,0],dataB_q2.iloc[:,1], c='b', marker="_", label='dataB_q2')
    plt.legend(loc='lower left')
    plt.show()
    
def q2b():
    
    X = q2_data.iloc[:, :2].values
    y = q2_data.iloc[:, -1].values
    
    svm_model_1 = svm.SVC(kernel='linear',C=0.1).fit(X, y) 
    svm_model_2 = svm.SVC(kernel='linear',C=1).fit(X, y) 
    svm_model_3 =  svm.SVC(kernel='linear',C=10).fit(X, y) 
    svm_model_4 = svm.SVC(kernel='linear',C=100).fit(X, y) 
    
    plt.subplot(121)
    plt.clf()
    decision_boundary(svm_model_1, X.min(), X.max())
    plt.scatter(dataA_q2.iloc[:,0],dataA_q2.iloc[:,1], c='g', marker="+", label='dataA_q2')
    plt.scatter(dataB_q2.iloc[:,0],dataB_q2.iloc[:,1], c='b', marker="_", label='dataB_q2')
    plt.title('Linear SVM - C:0.1')
    plt.axis([180, 440, 0, 350])
    plt.legend(loc='upper left')
    plt.show()
    
    
    plt.subplot(122)
    plt.clf()
    decision_boundary(svm_model_2, X.min(), X.max())
    plt.scatter(dataA_q2.iloc[:,0],dataA_q2.iloc[:,1], c='g', marker="+", label='dataA_q2')
    plt.scatter(dataB_q2.iloc[:,0],dataB_q2.iloc[:,1], c='b', marker="_", label='dataB_q2')
    plt.title('Linear SVM - C:1')
    plt.legend(loc='upper left')
    plt.axis([180, 440, 0, 350])
    plt.show()
    
    
    plt.subplot(221)
    plt.clf()
    decision_boundary(svm_model_3, X.min(), X.max())
    plt.scatter(dataA_q2.iloc[:,0],dataA_q2.iloc[:,1], c='g', marker="+", label='dataA_q2')
    plt.scatter(dataB_q2.iloc[:,0],dataB_q2.iloc[:,1], c='b', marker="_", label='dataB_q2')
    plt.title('Linear SVM - C:10')
    plt.axis([180, 440, 0, 350])
    plt.legend(loc='upper left')
    plt.show()
   
    
    plt.subplot(222)
    plt.clf()
    decision_boundary(svm_model_4, X.min(), X.max())
    plt.scatter(dataA_q2.iloc[:,0],dataA_q2.iloc[:,1], c='g', marker="+", label='dataA_q2')
    plt.scatter(dataB_q2.iloc[:,0],dataB_q2.iloc[:,1], c='b', marker="_", label='dataB_q2')
    plt.title('Linear SVM - C:100')
    plt.axis([180, 440, 0, 350])
    plt.legend(loc='upper left')
    plt.show()
    plt.clf()

#function to implement adaboost m1 algorithm from weak SVM classifiers
def q2c(X,y,X_test,penalty):
      T = 0
      
      ensemble_pred =list()
      weak = svm.SVC(kernel='linear',C=penalty)
      initial_weights = np.ones(X.shape[0])*1/(X.shape[0])
      weights_Di  = np.array(initial_weights)
      weights_Di = np.reshape(weights_Di,newshape=(X.shape[0],1))
      X['Label'] = y
      X['Weights_Di'] = weights_Di
    
      class1 = np.zeros(X_test.shape[0])
      class2 = np.zeros(X_test.shape[0])
      while(T<50):
        
        # Randomly selecting 100 samples
        sample = X.sample(frac=0.277,replace=True) 
        weak_X = sample.iloc[:,1:3]
        weak_y = sample.iloc[:,3]
        weak_val = X.iloc[:,1:3]
        # Train the 100 samples
        weak.fit(weak_X, weak_y,sample_weight=sample.iloc[:,4]) 
        #Predict all samples 
        weak_prediction = weak.predict(weak_val)
        error = 0
        train_len = X.shape[0]
        #calculating error in the predictions
        for jj in range(train_len):
            if(X.iloc[jj,3]!=weak_prediction[jj]):
                error = error + X.iloc[jj,4]
        if (error > 0.5): 
          #when error is greater than 50% sample the data and train other classifier.
          continue
        beta_value = error / (1 - error)
        # damping weights of correctly classified samples
        for kk in range(train_len):
            if(X.iloc[kk,3]==weak_prediction[kk]):
                X.iloc[kk,4] = X.iloc[kk,4]*beta_value
        normalisation  = np.sum(X.iloc[:,4])
        X.iloc[:,4] = X.iloc[:,4].div(normalisation)
        weak_test = X_test.iloc[:,0:2]
        test_pred = weak.predict(weak_test)
        res = np.log(1/beta_value)
        for yy in range(len(test_pred)):
            if(test_pred[yy]==0):
            
                class1[yy] = np.add(class1[yy],res)
            else:
                class2[yy] = np.add(class2[yy],res)
    
        T= T+1
      
      for zz in range(len(class1)):
          if(class2[zz]>class1[zz]):
              pred = 1
          else:
              pred=0
          ensemble_pred.append(pred)
      
    
      return np.array(ensemble_pred)
 
    
def boundary_ensemble(plot,q2_data,x_grid, y_grid, **params):
  
    X =  q2_data.iloc[:,0:2]
    y = q2_data['Label'].values
    X_test = pd.DataFrame(np.c_[x_grid.ravel(), y_grid.ravel()])
    X = X.reset_index()
    ada_M1 = q2c(X,y,X_test,100)
    ada_M1 = ada_M1.reshape(x_grid.shape)
    contour = plot.contourf(x_grid, y_grid, ada_M1, **params)
    
    return contour


def q2_accuracies():
    accuracies1 = list()
    accuracies2 = list()
    accuracies3 = list()
    accuracies4 = list()
    
    X = q2_data.iloc[:, :2].values
    y = q2_data.iloc[:, -1].values
    svm_model1 = svm.SVC(kernel='linear', C=0.1)
    svm_model2 = svm.SVC(kernel='linear', C=1)
    svm_model3 = svm.SVC(kernel='linear',C=10)
    svm_model4 = svm.SVC(kernel='linear',C=100)
    for tt in range(10):
        scores1 = cross_val_score(svm_model1, X, y, cv=10)
        accuracies1.append(np.mean(scores1))
        scores2 = cross_val_score(svm_model2, X, y, cv=10)
        accuracies2.append(np.mean(scores2))
        scores3 = cross_val_score(svm_model3, X, y, cv=10)
        accuracies3.append(np.mean(scores3))
        scores4 = cross_val_score(svm_model4, X, y, cv=10)
        accuracies4.append(np.mean(scores4))
    print('10 times 10 fold Accuarcy for SVM: C=0.1 ', np.mean(accuracies1))
    print('10 times 10 fold Accuarcy for SVM: C=1 ', np.mean(accuracies2))
    print('10 times 10 fold Accuarcy for SVM: C=10 ', np.mean(accuracies3))
    print('10 times 10 fold Accuarcy for SVM: C=100 ', np.mean(accuracies4))
    
    
def q2d():    
    accuracy_ADA =list()
    X = q2_data.iloc[:, :2]
    y = q2_data.iloc[:, 2]
    ten_x_ten = RepeatedKFold(n_splits=10, n_repeats=10)
    for train_loc, test_loc in ten_x_ten.split(X, y):
      X_t, X_test = q2_data.loc[train_loc], q2_data.loc[test_loc]
      y_t, y_test = q2_data.loc[train_loc]['Label'].values, q2_data.loc[test_loc]['Label'].values
      X_t = X_t.reset_index()
      ADA_res =q2c(X_t,y_t,X_test,0.1)
      acc=accuracy_score(y_test,ADA_res)
      accuracy_ADA.append(acc)
    print("The Mean accuracy is:",np.mean(accuracy_ADA))
    print("The Variance of Accuracies is:",np.var(accuracy_ADA))
    
    
def q2e():
    X = q2_data.iloc[:, :2].values
    y = q2_data.iloc[:,2].values
    fig,ax = plt.subplots(1, 1)
    plt.subplots_adjust(wspace=0.8, hspace=0.8)
    comp1, comp2 = X[:, 0], X[:, 1]
    x, Y = boundary_mesh(comp1, comp2) 
    boundary_ensemble(ax,q2_data,x,Y,cmap=plt.cm.winter, alpha=0.8)
    plt.scatter(comp1, comp2, c=y, cmap=plt.cm.winter, s=20, edgecolors='k')
    plt.xlabel('1st feature')
    plt.ylabel('2nd feature')
    plt.title('ADABOOST M1 Decision Boundary SVM with C =100')
    plt.show()

def main():  
    print("Visualising data for question 1")
    q1a()
    q1b_c()


    print("Done with question 1, Starting Question 2")
    q2a()
    q2b()
    q2_accuracies()
    q2d()
    q2e()
       
if __name__ == "__main__":
    main()

