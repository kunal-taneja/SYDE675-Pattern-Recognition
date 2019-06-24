# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:14:23 2019

@author: Kunal Taneja
"""

#importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat

#function to meet requirements of Q1 part a
def Q1a(dataA,dataB): 
    
    #plotting scatter plot which takes both feature values as an argument
    plt.subplot(121,aspect='equal')
    plt.scatter(dataA.T[:,0], dataA.T[:,1], c='blue')
    plt.title('data generated for class A')
   
    plt.subplot(122,aspect='equal')
    plt.scatter(dataB.T[:,0], dataB.T[:,1], c='yellow')
    plt.title('data generated for class B')
    
    plt.show()
    
def Q1b(dataA,dataB):
    
    #using inbuild function to evaluate eigenvalues and eigenvectors corresponding to covariance matrix of class A and B.
    eigValA, eigVecA = np.linalg.eig(np.cov(dataA))
    eigValB, eigVecB = np.linalg.eig(np.cov(dataB))
    """
    1. Calculating first stddev contour making use of eigenvalues and eigenvectos.
    2. Used ellipse method defined in patches library and aligned prinicipal axes in the direction of eigenvectors.
    3. Magnitude of axes is set as square root of eigenvalues.
    4. Height and width parameter take the diameter of prinicipal axes so multiplying the sqrt of eigenvalues by 2.
    5. Finally angle is alligned with eigenvectors and contour is made over the scatter plot.
    6. As the mean is zero(as mentioned in question) the centre of ellipse is set as (0,0).
    """
    ax = plt.subplot(121,aspect='equal') 
    contour = pat.Ellipse(xy=(0,0),width=np.sqrt(eigValA[0])*2,height=np.sqrt(eigValA[1])*2,angle=np.rad2deg(np.arccos(eigVecA[0,0])),color='black')
    contour.set_facecolor('black') 
    contour.set_alpha(0.3)
    ax.add_artist(contour)      
    plt.scatter(dataA.T[:,0], dataA.T[:,1], c='blue')
    plt.title('first stddev contour class A')
    
    
    ax = plt.subplot(122,aspect='equal') 
    contour = pat.Ellipse(xy=(0,0),width=np.sqrt(eigValB[0])*2,height=np.sqrt(eigValB[1])*2,angle=np.rad2deg(np.arccos(eigVecB[0,0])),color='black')
    contour.set_facecolor('black') 
    contour.set_alpha(0.3)
    ax.add_artist(contour)      
    plt.scatter(dataB.T[:,0], dataB.T[:,1], c='yellow')
    plt.title('first stddev contour class B')
    
    plt.show()
    
def Q1c_d(dataA,dataB,cov_A,cov_B):
    
    #calculating covariance for class A using formula.
    # cov(x,y) = summation{(x-mean(x))*(y-mean(y))} / n-1
    submeanA = dataA.T - [dataA.T[:,0].mean(), dataA.T[:,1].mean()]
    covA = np.matmul(submeanA.T,submeanA)/(1000-1)
    #comparing calculated and expected covariance matrix.
    print('Calculated sample covariance of dataset A is : ', list(covA) )
    print('Expected covariance matrix of class A is:',cov_A)
    
    #calculating covariance for class B using formula.
    # cov(x,y) = summation{(x-mean(x))*(y-mean(y))} / n-1
    submeanB = dataB.T - [dataB.T[:,0].mean(), dataB.T[:,1].mean()]
    covB = np.matmul(submeanB.T,submeanB)/(1000-1)
    #comparing calculated and expected covariance matrix.
    print('Calculated sample covariance of dataset B is : ', list(covB) )
    print('Expected covariance matrix of class B is:',cov_B)
    
    
    
def main():
    #given covariance matrix for class A and B
     cov_A = [[1,0],[0,1]]
     cov_B = [[1,0.9],[0.9,1]]
    
    #performing cholesky decomposition of the covariance matrix of A and B.
     chol_A = np.linalg.cholesky(cov_A)
     chol_B = np.linalg.cholesky(cov_B)
     
    #generating 1000 data samples corresponding to both features. 
     data = np.random.standard_normal(size=(1000,2))
    
    #dataA and dataB are generated based on the cholesky decomposition for class A and class B.
     dataA = np.dot(chol_A,data.T)
     dataB = np.dot(chol_B,data.T)
     Q1a(dataA,dataB)
     Q1b(dataA,dataB)
     Q1c_d(dataA,dataB,cov_A,cov_B)
    
if __name__== "__main__":
  main()

