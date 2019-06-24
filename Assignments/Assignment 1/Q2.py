# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:46:59 2019

@author: Kunal Taneja
"""
#importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from sklearn.metrics import confusion_matrix
from numpy.linalg import norm, inv
from matplotlib.colors import ListedColormap


#Function to generate decision boundaries corresponding to MAP and ML rule.
def decision_boundaries():
    #Given Covariance of the classes
    cov_A= np.array([[1, -1],[-1, 2]])
    cov_B = np.array([[1, -1],[-1, 7]])
    cov_C = np.array([[0.5, 0.5],[0.5, 3]])
    
    #using inbuild function to evaluate eigenvalues and eigenvectors corresponding to covariance matrix of class A,B and C.
    eigValA, EigVecA = np.linalg.eig(cov_A)
    eigValB, EigVecB = np.linalg.eig(cov_B)
    eigValC, EigVecC = np.linalg.eig(cov_C)
    
   
    
    colours = ListedColormap(['yellow', 'blue', 'orange'])
    labels = ['CLass A' , 'CLass B' , 'CLass C']
    #using grid sampling to classify points in the grid and color them with appropriate class color defined in colormap.
    x_min, x_max = 0, 10
    y_min, y_max = 0, 10
    
    # step size of 0.5 taken in the meshgrid
    step = .05 
    
    #making  a grid of size 10x10 using np.arrange function in numpy
    X, Y = np.meshgrid(np.arange(x_min, x_max, step),np.arange(y_min, y_max, step))

    #stroing results of MAP and ML classifiers in 1D vectors
    MAP_result= MAP_class(np.c_[X.ravel(), Y.ravel()])
    ML_result = ML_class(np.c_[X.ravel(), Y.ravel()])


    #Reshaping the results so that they can be used in plt.contour method
    MAP_result = np.reshape(MAP_result,X.shape)
    ML_result = np.reshape(ML_result,X.shape)

    #Using Ellipse method to plot first stddev contours similar to what did in Q1.
    ax = plt.subplot(121,aspect='equal')
    contour1 = pat.Ellipse(xy=(3,2),width=np.sqrt(eigValA[0])*2,height=np.sqrt(eigValA[1])*2,angle=np.rad2deg(np.arccos(EigVecA[0,0])),color='#FFAAAA')
    contour2 = pat.Ellipse(xy=(5,4),width=np.sqrt(eigValB[0])*2,height=np.sqrt(eigValB[1])*2,angle=np.rad2deg(np.arccos(EigVecB[0,0])),color='#AAFFAA')
    contour3 = pat.Ellipse(xy=(2,5),width=np.sqrt(eigValC[0])*2,height=np.sqrt(eigValC[1])*2,angle=np.rad2deg(np.arccos(EigVecC[0,0])),color='#AAAAFF')
    contour1.set_facecolor('black') 
    contour1.set_alpha(0.3)
    ax.add_artist(contour1)
    contour2.set_facecolor('black') 
    contour2.set_alpha(0.3)
    ax.add_artist(contour2)
    contour3.set_facecolor('black') 
    contour3.set_alpha(0.3)
    ax.add_artist(contour3)
    
    #plotting the colored datasamples over the grid which would help in visulaising the decision boundaries.
    plt.pcolormesh(X, Y, MAP_result, cmap=colours)
    plt.contour(X, Y, MAP_result, colors='black')
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title("Decision Boundary for MAP")
    
    
    #Doing same which ML decision boundaries.
    ax = plt.subplot(122,aspect='equal')
    contour1 = pat.Ellipse(xy=(3,2),width=np.sqrt(eigValA[0])*2,height=np.sqrt(eigValA[1])*2,angle=np.rad2deg(np.arccos(EigVecA[0,0])),color='#FFAAAA')
    contour2 = pat.Ellipse(xy=(5,4),width=np.sqrt(eigValB[0])*2,height=np.sqrt(eigValB[1])*2,angle=np.rad2deg(np.arccos(EigVecB[0,0])),color='#AAFFAA')
    contour3 = pat.Ellipse(xy=(2,5),width=np.sqrt(eigValC[0])*2,height=np.sqrt(eigValC[1])*2,angle=np.rad2deg(np.arccos(EigVecC[0,0])),color='#AAAAFF')
    contour1.set_facecolor('black') 
    contour1.set_alpha(0.3)
    contour1.set_label('Class A')
    ax.add_artist(contour1)
    contour2.set_facecolor('black') 
    contour2.set_alpha(0.3)
    contour1.set_label('Class B')
    ax.add_artist(contour2)
    contour3.set_facecolor('black') 
    contour3.set_alpha(0.3)
    contour1.set_label('Class C')
    ax.add_artist(contour3)
    plt.pcolormesh(X, Y, ML_result, cmap=colours,label=labels)
    plt.contour(X, Y , ML_result, colors='black')
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title("Decision Boundary for ML")
    plt.legend
    plt.show()

    return





#Function to create MAP classifier
def MAP_class(values):
    
    p = list()
    
    #Given Covariance of the classes
    cov_A= np.array([[1, -1],[-1, 2]])
    cov_B = np.array([[1, -1],[-1, 7]])
    cov_C = np.array([[0.5, 0.5],[0.5, 3]])
    
    #Given Mean of the Classes
    u1= np.array([3,2])
    u2= np.array([5,4])
    u3= np.array([2,5])
    
    #Given prior probabilities
    P_A = 0.2
    P_B = 0.3
    P_C = 0.5

    #Using the mathetical eqn for MAP rule provided in lectures for classifying data point based on the scores we get.  
    for x in values:
        score1 = (-0.5*np.dot(np.dot((x - u1),inv(cov_A)),(x - u1).T) - 0.5 * np.log(norm(cov_A)) + np.log(P_A))
        score2 = (-0.5*np.dot(np.dot((x - u2),inv(cov_B)),(x - u2).T) - 0.5 * np.log(norm(cov_B)) + np.log(P_B))
        score3 = (-0.5*np.dot(np.dot((x - u3),inv(cov_C)),(x - u3).T) - 0.5 * np.log(norm(cov_C)) + np.log(P_C))
    
    #after finding score for each class we concatenate the scores and find the index which had maximum score.
    #That is index = 0 means Class A had maximum score so sample belongs to Class A
        score = [score1, score2, score3]
        p.append(np.argmax(score))
    
    return p

#Function to create ML classifier
def ML_class(values):
    
    p = list()
    
    #Given Covariance of the classes
    cov_A= np.array([[1, -1],[-1, 2]])
    cov_B = np.array([[1, -1],[-1, 7]])
    cov_C = np.array([[0.5, 0.5],[0.5, 3]])
    
    #Given Mean of the Classes
    u1= np.array([3,2])
    u2= np.array([5,4])
    u3= np.array([2,5])

    #Using the mathetical eqn for ML rule provided in lectures for classifying data point based on the scores we get.  
    for x in values:
        score1 = (-0.5*np.dot(np.dot((x - u1),inv(cov_A)),(x - u1).T) - 0.5 * np.log(norm(cov_A)))
        score2 = (-0.5*np.dot(np.dot((x - u2),inv(cov_B)),(x - u2).T) - 0.5 * np.log(norm(cov_B)))
        score3 = (-0.5*np.dot(np.dot((x - u3),inv(cov_C)),(x - u3).T) - 0.5 * np.log(norm(cov_C)))
    
    #after finding score for each class we concatenate the scores and find the index which had maximum score.
    #That is index = 0 means Class A had maximum score so sample belongs to Class A
        score = [score1, score2, score3]
        p.append(np.argmax(score))
    
    return p





def main():
    
    #calling decision boundary method to see what it looks like.
    decision_boundaries()
    
    
    #Generate 3000 Samples corresponding to what we did in Q1 
    classA = np.random.multivariate_normal([3,2], [[1, -1],[-1, 2]], 600)
    classB = np.random.multivariate_normal([5,4], [[1, -1],[-1, 7]], 900) 
    classC = np.random.multivariate_normal([2,5], [[0.5, 0.5],[0.5, 3]], 1500)

    #As generating data doesnt generate class labels so we explicitly generate class labels.
    label_A = [0] * 600  #P_A = 0.2 and 0.2 of 3000 = 600
    label_B = [1] * 900  #P_B = 0.3
    label_C = [2] * 1500 #P_C = 0.5
    #Combine all predicted data into one list
    true_data = np.concatenate((label_A, label_B, label_C), axis=0)


    #Combine all 3000 data into one list
    given_data =np.concatenate((classA, classB, classC), axis=0)

    #Classify using map and ml classifier function
    map_res = MAP_class(given_data)
    ml_res = ML_class(given_data)

    #using builtin methods to evaluate the confusion matrix
    print('Confusion Matrix MAP:')
    conf_MAP = confusion_matrix(true_data, map_res)
    print(conf_MAP)
    print('Confusion Matrix of ML Classifier:')
    conf_ML = confusion_matrix(true_data, ml_res)
    print(conf_ML)
    
    #calculating classification error for each class and printing
    print('P(Err) MAP Classifier For:\n Class A- {},\n Class B- {},\n Class C- {}'.format((conf_MAP[0][1]+conf_MAP[0][2])/3000,(conf_MAP[1][0]+conf_MAP[1][2])/3000,(conf_MAP[2][0]+conf_MAP[2][1])/3000))
    print('P(Err) ML Classifier For:\n Class A- {},\n Class B- {},\n Class C- {}'.format((conf_ML[0][1]+conf_ML[0][2])/3000,(conf_ML[1][0]+conf_ML[1][2])/3000,(conf_ML[2][0]+conf_ML[2][1])/3000))
    

if __name__== "__main__":
  main()