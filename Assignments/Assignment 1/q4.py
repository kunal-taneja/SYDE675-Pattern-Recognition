
#importing required libraries
import numpy as np
import pandas as pd
from mlxtend.data import loadlocal_mnist
from scipy.spatial import distance
from collections import Counter
from sklearn.metrics import accuracy_score


#used loadlocal_mnist library to load the dataset.
train_data, train_label = loadlocal_mnist(
        images_path='train-images.idx3-ubyte', 
        labels_path='train-labels.idx1-ubyte')
test_data, test_label = loadlocal_mnist(
        images_path='t10k-images.idx3-ubyte', 
        labels_path='t10k-labels.idx1-ubyte')
 

def fit_method(train_data, train_label, test_data, k,nearest_dist):
    labels = list()
    #using argpartition method to select k neighbours.
    neighbor = np.argpartition(nearest_dist, k)[:k]
    # makes a list of the k neighbors along with their label values
    for i in neighbor:
        labels.append(train_label[i])

    # return most common target
    return Counter(labels).most_common(1)[0][0]

def Q4a(train_data, train_label, test_data, ans, k,nearest_dist):
    #using fit_method to make predictions for the dataset.
    for i in range(len(test_data)):
        ans.append(fit_method(train_data, train_label, test_data[i, :], k,nearest_dist[i,:]))
    return ans


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

def Q4b():
    d = [5,50,100,500]
    k = [1,3,5,11]
    res_acc = list()
    PCA_D = np.array(train_data.T)
    PCA_Test = np.array(test_data.T)
    for i in d:
        pca_data,d,matrix_w,eig_val_sorted = Q3a_b(PCA_D,5)
     
        pca_test,d,matrix_w,eig_val_sorted = Q3a_b(PCA_Test,5)
    
        distance1 = distance.cdist(pca_test.T, pca_data.T, 'euclidean')
        for j in k:
                ans = list()
                output = Q4a(pca_data.T,train_label , pca_test.T, ans,j,distance1)
                ans = np.asarray(output)
                # calculating accuracy for the predictions obtained
                accuracy = accuracy_score(test_label, ans)
                print(accuracy)
                res_acc.append(accuracy)


def main():
#using euclidean method to find distance between the sample and their labels.
    nearest_dist = distance.cdist(test_data, train_data, 'euclidean')
    
    
    k = [1,3,5,11]
    res_acc = list()
    for i in k:
        ans = list()
        result = Q4a(train_data, train_label, test_data, ans,i,nearest_dist)
        ans = np.asarray(result)
        # calculating accuracy for the predictions obtained
        accuracy = accuracy_score(test_label, ans)
        res_acc.append(accuracy)
        
#storing and showing the results in the form of dataframe
    list1= [res_acc[0], res_acc[1], res_acc[2], res_acc[3]]
    list2 = ['k_1','k_3','k_5','k_11']
    res=pd.DataFrame(list1)
    res.columns=["Accuracy"]
    res.index = list2
    print(res)
    
    Q4b()


if __name__== "__main__":
  main()


