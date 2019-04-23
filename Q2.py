from collections import Counter
from itertools import combinations
from binary_classifier_svm import *
import pickle
from sklearn.model_selection import train_test_split

class multidigit_gaussian_classifier:

    def __init__(self,file,gamma):
        self.file = file
        self.gamma = gamma
     
    def dump_object(self,obj,file):
        # open the file for writing
        obj_writer = open(file,'wb') 
        pickle.dump(obj,obj_writer)   
        obj_writer.close()
        print('duming done:',file)
        
    def get_the_object(self,file):
        # we open the file for reading
        obj_reader = open(file,'rb')  
        obj = pickle.load(obj_reader)  
        obj_reader.close()
        return obj
        
    def get_data(self,file):
        data = panda.read_csv(file, header = None)
        data = np.array(data.values)

        features = data[:,:-1]
        features = features / 255
        labels = data[:,-1]
        return features,labels 
       
    def get_max_scoring_elem(self,prediction_list,score):
        data = Counter(prediction_list)
        unique_freq_pair = data.most_common()   # Returns all unique items and their counts
        elem,max_freq = data.most_common(1)[0]
        
        max_occuring_elems = [elem for (elem,freq) in unique_freq_pair if freq == max_freq]
        
        scores = np.array([np.sum(score[prediction_list == elem]) for elem in max_occuring_elems])
        return max_occuring_elems[np.argmax(scores)]
  
    def set_model(self):
        
        features,labels = self.get_data(self.file)
        self.features = features
        self.labels = labels
        
        unique_labels = np.unique(self.labels)
        digit_pairs = list(combinations(unique_labels, 2))
        
        models = []
        count = 0
        for pair in digit_pairs:
            count += 1
            print(count)
            digit0,digit1 = pair
            filter_cond = (self.labels == digit0) | (self.labels == digit1)
            
            features_req = self.features[filter_cond]
            labels_req = self.labels[filter_cond]
            labels_req = np.where(labels_req == digit0, 1, -1)
            
            svm = SVM('gaussian',self.gamma)
            svm.set_model(features_req,labels_req,digit0,digit1)
            w,b = svm.model  #w=1 is dummy
            models.append(svm)
            
        self.models_digitpairs = list(zip(models,digit_pairs))
       
        return models
    
    def set_predictions(self,test_file):
        
        test_features,test_labels = self.get_data(test_file)
        self.test_features = test_features
        self.test_labels = test_labels
        
        predictions = []
        prediction_scores = []
        
        for svm,pair in self.models_digitpairs:
            digit0,digit1 = pair
            w,b = svm.model
            prediction = svm.get_predictions(self.test_features,w,b)
            
            prediction_scores.append(np.absolute(prediction))
            
            prediction[prediction >= 0] = digit0
            prediction[prediction < 0] = digit1
            
            predictions.append(prediction)
      
        self.predictions = np.array(predictions).T
        self.prediction_scores = np.array(prediction_scores).T
        
    def multi_class_accuracy(self):
        prediction = []
        for prediction_list,prediction_score in zip(self.predictions,self.prediction_scores):
            predicted_val = self.get_max_scoring_elem(prediction_list,prediction_score)
            prediction.append(predicted_val)    
        self.accuracy = accuracy_score(self.test_labels,prediction)
        print(confusion_matrix(self.test_labels,prediction))
     
    def lib_svm(self,train_file,test_file):
        features, labels = self.get_data(train_file)
        training_data = svm_problem(labels, features)
        params = svm_parameter('-s 0 -t 2 -c 1 -g 0.05')
        
        model = svm_train(training_data, params)
        
        test_features, test_labels = self.get_data(test_file)
        p_labels, p_acc, p_vals = svm_predict(test_labels, test_features, model)
        

train_file = "../mnist/train.csv"
test_file = "../mnist/test.csv"
classifier = multidigit_gaussian_classifier(train_file,0.05)
models = classifier.set_model()

classifier.dump_object(models,"multi_class_digit_classification")
classifier.set_predictions(test_file)
classifier.multi_class_accuracy()

print("Accuracy:",classifier.accuracy,"%")

print(classifier.lib_svm(train_file,test_file))

def get_best_hyper_parameter(self,C_list,train_file,test_file):  
    features, labels = self.get_data(train_file)
    test_features, test_labels = self.get_data(test_file)
    features,vld_features,labels,vld_labels = train_test_split(features,labels,test_size=0.1,random_state=98)

    max_acc = 0
    best_c = C_list[0]
    validation_accuracy = []
    test_accuracy = []
    for c in C_list:
        print(c)
        training_data = svm_problem(labels, features)
        arg = '-s 0 -t 2 -c '+str(c)+' -g 0.05'
        params = svm_parameter(arg)
        model = svm_train(training_data, params)
        v_labels, v_acc, v_vals = svm_predict(vld_labels, vld_features, model)
        t_labels, t_acc, t_vals = svm_predict(test_labels, test_features, model)
        
        validation_accuracy.append(v_acc[0])
        test_accuracy.append(t_acc[0])
        
        accuracy = t_acc[0]
        print(accuracy)
        if(max_acc < accuracy):
            max_acc = accuracy
            best_c = c
    return best_c,validation_accuracy,test_accuracy


C_list = [1e-5,1e-3,1,5,10]
best_c,validation_accuracy,test_accuracy = get_best_hyper_parameter(classifier,C_list,train_file,test_file)

import matplotlib.pyplot as plt
def plot(cs,test_acc,val_acc):
    fig = plt.figure()
    plt.xlabel("Log(c)")
    plt.ylabel("Accuracy")
    plt.plot(cs,test_acc,label = 'Test Set Accuracy')
    plt.plot(cs,val_acc,label = 'Validation Set Accuracy')
    plt.legend(loc = 'upper left')
C_list = np.log(np.array([1e-5,1e-3,1,5,10]))
plot(C_list,test_accuracy,validation_accuracy)


