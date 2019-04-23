import utils as utility
import numpy as np
import pickle
import math
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from random import randint
import os
import sys

def nltk2word_tag(nltk_tag):
    starts_with = ['J','V','N','R']
    tags = [wordnet.ADJ,wordnet.VERB,wordnet.NOUN,wordnet.ADV,None]
    ind = starts_with.index(nltk_tag[0]) if nltk_tag[0] in starts_with else 4
    return tags[ind]

def lemmatize(sentence):
    lemmatizer = WordNetLemmatizer()
    en_stop = set(stopwords.words('english'))
    tokens = word_tokenize(sentence.lower())
    for token in tokens:
        if token in en_stop:
            tokens.remove(token)
    
    nltk_tagged = nltk.pos_tag(tokens)
  
    word_tag = map(lambda x: (x[0], nltk2word_tag(x[1])), nltk_tagged)
    lemm_words = []
    for word, tag in word_tag:
        if tag is None:            
            lemm_words.append(word)
        else:
            lemm_words.append(lemmatizer.lemmatize(word, tag))
    return lemm_words

def get_ngrams(input_words,n):
    args = [input_words[i:] for i in range(n)]
    return list(zip(*args))

def feature_engg(words,feature='None'):
    if(feature == 'bigram'):
        return get_ngrams(words,2)
    if(feature == 'trigram'):
        return get_ngrams(words,3)
    return words

def text_processing(text,type='None'):
    if(type=='stemming'):
        return utility.getStemmedDocuments(text)
    if(type == 'lemmatize'):
        return lemmatize(text)
    return text.split()

def set_model(train_file,model_file,preprocess_type='None',feature='None'):

    if os.path.exists(model_file):
        return

    docs =  utility.json_reader(train_file)
    stars = np.zeros(5)
    category_count = np.zeros(5)

    class_frequency = {}
    count = 0
    all_words = []
    
    for doc in docs:
        count = count + 1
        if(count%1000 == 0):
            print(count)
        words = text_processing(doc['text'],preprocess_type)
        words = feature_engg(words,feature)
            
        star = int(doc['stars'])
        stars[star-1] += 1
        category_count[star-1] += len(words)

        for word in words:
            if word not in class_frequency:
                all_words.append(word)
                class_frequency[word] = np.ones(5)
            class_frequency[word][star-1] += 1

    #print(class_frequency)
    #print(all_words)

    m = count
    vocab_size = len(all_words)
    category_count += vocab_size

    for i in all_words:
        class_frequency[i] = np.log(class_frequency[i]/category_count)
   
    phai_y = np.log(stars/m)
 
    parameters = [class_frequency,phai_y,category_count]
    
    # open the file for writing
    obj_writer = open(model_file,'wb') 
    pickle.dump(parameters,obj_writer)   
    obj_writer.close()
    print('done')

def get_the_model(model_file):
    # we open the file for reading
    obj_reader = open(model_file,'rb')  
    model = pickle.load(obj_reader)  
    obj_reader.close()
    return model


def get_prediction(test_file,model_file,mode='None',preprocess_type='None',feature='None'):
    parameters = get_the_model(model_file)
    
    count = 0 
    prob_dict = parameters[0]  
    phai_y=parameters[1]
    category_count = parameters[2]
    print(len(prob_dict))
    
    docs =  utility.json_reader(test_file)
    prediction =[]
    original = []
    
    for doc in docs:
        if(count%100000 == 0):
            print("iter:",count)
        count += 1

        if mode == 'b1':
            prediction.append(randint(1,5))
        elif mode == 'b2':
            prediction.append(np.argmax(category_count)+1)
        elif mode == 'a':
            words = text_processing(doc['text'],preprocess_type)
            words = feature_engg(words,feature)

            sum_of_logs = phai_y
            for word in words:
                if word not in prob_dict:
                    sum_of_logs = np.add(sum_of_logs,np.log(1/category_count))
                else:
                    sum_of_logs = np.add(sum_of_logs,prob_dict[word])

            prediction.append(np.argmax(sum_of_logs)+1)
        original.append(int(doc['stars']))
       
    return prediction,original

def compute_testdata_accuracy(prediction,original):
    return accuracy_score(original,prediction)

def get_confusion_matrix(prediction,original):
    return confusion_matrix(original,prediction)

def get_f1score_macro(prediction,original):
    return f1_score(original,prediction,average='macro')

def get_f1score(prediction,original):
    return f1_score(original,prediction,average=None)

if __name__ == '__main__':
    #reading the data from files
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    mode = sys.argv[3]

    if mode == 'a':
        model_file = "prob_noextrafilter"
        set_model(train_file,model_file)
        prediction,original = get_prediction(train_file,model_file,mode)
        print("Train data accuracy: ",compute_testdata_accuracy(prediction,original))

        prediction,original = get_prediction(test_file,model_file,mode)
        print("Test data accuracy: ",compute_testdata_accuracy(prediction,original))
    elif mode == 'b':
        model_file = "prob_noextrafilter"
        set_model(train_file,model_file)
        prediction,original = get_prediction(test_file,model_file,'b1')
        print("Test data accuracy in Random Prediction: ",compute_testdata_accuracy(prediction,original))

        prediction,original = get_prediction(test_file,model_file,'b2')
        print("Test data accuracy in Majority Prediction: ",compute_testdata_accuracy(prediction,original))
    elif mode == 'c':
        model_file = "prob_noextrafilter"
        set_model(train_file,model_file)
        prediction,original = get_prediction(test_file,model_file,'a')

        print("Confusion Matrix = ",get_confusion_matrix(prediction,original))
    elif mode == 'd':
        model_file = "prob_stemming"
        set_model(train_file,model_file,'stemming')
        prediction,original = get_prediction(test_file,model_file,'a','stemming')
        print("Test data accuracy: ",compute_testdata_accuracy(prediction,original))

    elif mode =='e':
        model_file = "prob_stemming_bigram"
        set_model(train_file,model_file,'stemming','bigram')
        prediction,original = get_prediction(test_file,model_file,'a','stemming','bigram')
        print("Test data accuracy(stemming_bigram): ",compute_testdata_accuracy(prediction,original))

        model_file = "prob_lemmatize_bigram_with_stopwords"
        set_model(train_file,model_file,'lemmatize','bigram')
        prediction,original = get_prediction(test_file,model_file,'a','lemmatize','bigram')
        print("Test data accuracy(lemmatize_bigram): ",compute_testdata_accuracy(prediction,original))
    elif mode =='f':
        model_file = "prob_lemmatize_bigram_with_stopwords"
        set_model(train_file,model_file,'lemmatize','bigram')
        prediction,original = get_prediction(test_file,model_file,'a','lemmatize','bigram')
        print("macro F1 Score: ",get_f1score_macro(prediction,original))
        print("F1 Score: ",get_f1score(prediction,original))
    elif mode == 'g':
        model_file = "prob_best_model"
        set_model(train_file,model_file,'lemmatize','bigram')
        prediction,original = get_prediction(test_file,model_file,'a','lemmatize','bigram')
        print("Test data accuracy(lemmatize_bigram): ",compute_testdata_accuracy(prediction,original))
        print("macro F1 Score: ",get_f1score_macro(prediction,original))

