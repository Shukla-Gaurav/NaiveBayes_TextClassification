import pandas as panda 
import numpy as np
from cvxopt import matrix, solvers
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances

import sys
sys.path.insert(0,"/home/gaurav/Downloads/libsvm-3.23/python")

from svm import svm_parameter, svm_problem
from svmutil import svm_train, svm_predict
import pickle

class SVM:
    
    def __init__(self,kernel,gamma = 0.001275):
        self.kernel = kernel
        self.gamma = gamma

    def get_data(self,file,digit0,digit1):
        training_data = panda.read_csv(file, header = None)
        training_data = np.array(training_data.values)

        training_data_req = training_data[(training_data[:,-1] == digit0) | (training_data[:,-1] == digit1)]

        features = training_data_req[:,:-1]
        features = features / 255

        labels = training_data_req[:,-1]
        labels = np.where(labels == digit0, 1, -1)
        
        return features,labels
    
    def get_alphas(self,features,labels):  
        x_vals = features
        y_vals = labels.astype(float)
        
        m = x_vals.shape[0]
        K=1
        if self.kernel == 'gaussian':
            distance_sq = euclidean_distances(x_vals,x_vals, squared=True)
            K = np.exp( (-1)* distance_sq * self.gamma)
            P = np.outer(y_vals,y_vals) * K
            
        else:
            xdoty = y_vals[:, None] * x_vals
            P = np.dot(xdoty, xdoty.T)
            
        P = matrix(P)
        q = matrix(-np.ones((m, 1)))
        G = matrix(np.vstack((-np.eye(m),np.eye(m))))
        h = matrix(np.vstack((np.zeros((m,1)),np.ones((m,1)))))
        A = matrix(y_vals.reshape(1, -1))
        b = matrix(np.zeros(1))
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solution['x'])
        alphas = np.where(alphas>1e-4,alphas,0)
        return alphas,K

    def set_model(self,features,labels,digit0,digit1):  
        
        alphas,K = self.get_alphas(features,labels)
        print("SVs:",len([alpha for alpha in alphas if alpha>0]))
        
        c_alphas = alphas.reshape(-1)
        self.sv_alphas = alphas[c_alphas>0]
        self.sv_features = features[c_alphas>0]
        self.sv_labels = labels[c_alphas>0]
        
        if self.kernel == 'gaussian':
            sv_ind = np.where(c_alphas>0)[0]
            grid = np.ix_(sv_ind,sv_ind)
            sv_K = K[grid]
            b = self.sv_labels - np.sum(self.sv_alphas * self.sv_labels.reshape(self.sv_alphas.shape) * sv_K, axis = 0)    
            w = 1 #dummy 1 for w
        else:
            w = np.sum(self.sv_alphas * self.sv_labels.reshape(self.sv_alphas.shape) * self.sv_features, axis = 0)
            b = self.sv_labels - w @ self.sv_features.T
        
        b = np.mean(b) 
        self.model = (w,b)
    
    def get_predictions(self,test_features,w,b):
        if self.kernel == 'gaussian':
            distance_sq = euclidean_distances(self.sv_features,test_features, squared=True)
            K = np.exp( (-1)* distance_sq * self.gamma)
            prediction = np.sum(self.sv_alphas.reshape(-1,1) * self.sv_labels.reshape(self.sv_alphas.shape) * K, axis = 0) + b
        else:    
            prediction = w @ test_features.T + b

        return prediction
        
    def binary_classification_test(self,file,digit0,digit1):
        w,b = self.model
        print(w,b)
        test_features, test_labels = self.get_data(file,digit0,digit1)
        prediction = self.get_predictions(test_features,w,b)
        
        prediction[prediction >= 0] = 1
        prediction[prediction < 0] = -1
        
        self.accuracy = accuracy_score(test_labels,prediction)
    
    def lib_svm(self,train_file,test_file,digit0,digit1):
        features, labels = self.get_data(train_file,digit0,digit1)
        training_data = svm_problem(labels, features)
        
        if(self.kernel == 'gaussian'):
            params = svm_parameter('-s 0 -t 2 -c 1 -g 0.05')
        else:
            params = svm_parameter('-s 0 -t 2 -c 1 -g 0.001275')
            
        model = svm_train(training_data, params)
        
        test_features, test_labels = self.get_data(test_file,digit0,digit1)
        p_labels, p_acc, p_vals = svm_predict(test_labels, test_features, model)

if __name__ == '__main__':
    #reading the data from files
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    binary_multi = sys.argv[3]
    mode = sys.argv[4]

    if mode == 'a':
        digit0 = 1
        digit1 = 2
        svm = SVM('linear',0.0012)
        features,labels = svm.get_data(train_file,digit0,digit1)
        svm.set_model(features,labels,digit0,digit1)
        svm.binary_classification_test(test_file,digit0,digit1)
        print("Linear Accuracy:",svm.accuracy*100,"%")
    elif mode == 'b':
        digit0 = 1
        digit1 = 2
        svm = SVM('gaussian',0.05)
        features,labels = svm.get_data(train_file,digit0,digit1)
        svm.set_model(features,labels,digit0,digit1)
        svm.binary_classification_test(test_file,digit0,digit1)
        print("Gaussian Accuracy:",svm.accuracy*100,"%")
    elif mode == 'c':
        digit0 = 1
        digit1 = 2
        svm = SVM('linear',0.0012)
        svm.lib_svm(train_file,test_file,digit0,digit1)
        svm = SVM('gaussian',0.05)
        svm.lib_svm(train_file,test_file,digit0,digit1)        