# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:01:00 2019

@author: Kunal Taneja
"""
#importing required libraries
import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


#used loadlocal_mnist library to load the dataset.
train_data, train_label = loadlocal_mnist(
        images_path='train-images.idx3-ubyte', 
        labels_path='train-labels.idx1-ubyte')
test_data, test_label = loadlocal_mnist(
        images_path='t10k-images.idx3-ubyte', 
        labels_path='t10k-labels.idx1-ubyte')



#Function to implement PCA from scratch
def Q3a_b(X,n):
    #Calculating mean in order to calculate mean centered data.
    u = X.mean(axis=1)
    meansub = np.array(X.T) - u
    #Calculating the Covariance Matrix for the mean subtracted data
    cov = np.cov(meansub.T)
    #using inbuild function to evaluate eigenvalues and eigenvectors corresponding to covariance matrix
    EigVal, EigVec = np.linalg.eig(cov)
    eig_pairs = [(np.abs(EigVal[i]), EigVec[:,i]) for i in range(len(EigVec))]

    
    #Sorting eigenpairs in decreasing order so as to get principal components with greatest values.
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    
    #reshaping the eigpairs vector into a feature vector.
    reshaped = eig_pairs[0][1].reshape(784,1)
    for i in range(1,n):
        a = eig_pairs[i][1].reshape(784,1)     
        reshaped = np.concatenate((reshaped,a), axis=1)
    pca_data = reshaped.T.dot(meansub.T)
    #Choosing the Optimal value of D with POV Of 95%
    total = EigVal.sum()
    eig_val_sorted = sorted(EigVal, reverse=True)
    for i in range(0,784):
        pov = np.sum(eig_val_sorted[0:i]) / total
        if pov >= 0.95:
            d = i
            break
    return (pca_data,d,reshaped,eig_val_sorted)



#function plot show reconstruction results as d vs MSE plot
def Q3c(Data):
    
    mse_d_values = []
    d_values = [1,2,3,4,8,16,32,64,128,256,512,784]
   
    for i in d_values:
        pca_d,d,reshaped,eig_val_sorted = Q3a_b(Data,i)
        
        #reversing the PCA applied in order to get the reconsructed data
        reconstructed_data = reshaped.dot(pca_d)
        
        #using mean_squared_error library to calculate MSE for different values of d.
        mse = mean_squared_error(Data, reconstructed_data)
        mse_d_values.append(mse)
        
        
    #plotting d vs MSE plot    
    plt.plot(d_values,mse_d_values)
    plt.ylabel('MSE')
    plt.xlabel('d values')
    plt.xlim(0,794)
    plt.show()




#Function to disucss reconstruction over sample image '5' for different d values.
def Q3d(PCA_D):
    
    #Calculting mean in order to get mean centered data.
    u = train_data.mean(axis=1)
    meansub = np.array(train_data.T) - u
    DataNew = meansub.T
 
    fig, axs = plt.subplots(2, 3,figsize=(5,10))
    # Original Image from the Dataset
    first_image = (DataNew[0:1,:])
    first_label = train_label[0]
    # 784 columns correspond to 28x28 image
    plottable_image = np.reshape(first_image, (28, 28))
    # Plot the image
    c = 0
    r = 0
    axs[r][c].imshow(plottable_image, cmap='gray_r')
    axs[r][c].set_title('Original Image from Dataset Digit Label: {}'.format(first_label))
    ranges=[1,10,50,154,784]
    for i in ranges:
        pca_d,d,matrix_w,eig_val_sorted = Q3a_b(PCA_D,i)
        img = matrix_w.dot(pca_d[:,0:1]).T
        # First row is first image
        first_image = img
        first_label = train_label[0]
        # 784 columns correspond to 28x28 image
        plottable_image = (np.reshape(first_image, (28, 28)))
        if (c == 2):
            r = r + 1
            c = 0
        else:     
            c = c + 1
        # Plot the image
        axs[r][c].imshow(plottable_image, cmap='gray_r')
        axs[r][c].set_title('Digit Label:{} with PCA Reconstruction d = {}'.format(first_label,i))
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=2, hspace=0.25,
                        wspace=0.35)
    plt.show()
    plt.close()
    return eig_val_sorted


#this function uses sorted list of eigenvalues to make an eigenvalue vs d plot.
def Q3e(eig_val_sorted):
    
    #setting range of x-axis as range of d that is 1-784 
    x = list(range(1,785))
    eigsort = np.array(eig_val_sorted) / 100000

    plt.plot(x,eigsort)
    plt.ylabel('Eigen Values x 10^5')
    plt.xlabel('d values')
    plt.xlim(0,794)
    plt.show()

#entry to main    
def main():
    
    
    #Testing user defined PCA method build
    RawData = np.array(train_data.T)
    print('The Shape of data before applying PCA is:',RawData.shape)
    #PCA now takes X(DxN) and returns Y(dxN) where N is the number of samples, D is the number of input features, and d is the number of features selected by the PCA algorithm.
    pca_data,d,reshaped,eig_val_sorted = Q3a_b(RawData,10)
    print('The Shape of data after applying PCA with 10 components:',pca_data.shape)
    
    
    #value of d for POV = 95%
    print('\n')
    print('Value of d on POV of 95% is:',d)
   
    
    Q3c(RawData)
    eig = Q3d(RawData)
    Q3e(eig)


if __name__== "__main__":
  main()
