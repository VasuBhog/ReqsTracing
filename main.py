import numpy as np
from numpy import array
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt


#keras is a backend API for tensorflow
# import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Embedding
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

#supress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#START PROGRAM
#Convert input txt to csv
def txtToCsv(file,delimiter,csvFile):
    with open(file, 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        if file == 'input.txt':
            lines = (line.split(delimiter,1) for line in stripped if line)
        else: 
            lines = (line.split(delimiter) for line in stripped if line)
        with open(csvFile, 'w') as out_file:
            writer = csv.writer(out_file)
            if csvFile == 'input.csv':
                writer.writerow(('Title','Message'))
                writer.writerows(lines)
            else: 
                writer.writerow(('Title','NFR1','NFR2','NFR3'))
                writer.writerows(lines)



# inputdf = pd.read_fwf('input.txt', sep='', header=None)
# data.columns = ["a","b","c"]

if __name__ == "__main__":
    #set variables
    reqsCSV = 'input.csv'
    traceCSV = 'output.csv'
    #TESTING FILES 
    inputf = 'input.txt'
    outputf = 'output.txt'

    #Convert input file
    # inputf = input("Enter the input file: ")
    txtToCsv(inputf,':',reqsCSV)

    #Convert output fiile
    # outputf = input("Enter the output file: ")
    txtToCsv(outputf,',',traceCSV)

    #Create Dataframe 
    dfReqs = pd.read_csv(reqsCSV)
    dfTrace = pd.read_csv(traceCSV)

    #Merge the two dataframes/csvs into one
    df = pd.merge(dfReqs,dfTrace, on = 'Title')
    df.set_index('Title', inplace = True)
    print(df.shape)
    df.to_csv('Combined.csv')
    # df.info()
    # print(df.head())

    y = df[['NFR1', 'NFR2', 'NFR3']]
    # vectorizer = CountVectorizer()
    # text = df['Message']
    # vectorizer.fit(text)
    # print(vectorizer.vocabulary_)
    words = df['Message']
    print(words)
    max_length = max([len(s.split())for s in words])

    #TF-IDF 
    print(words)

    #Tokenize words
    t = Tokenizer(lower=True)                                      
    t.fit_on_texts(words)
    encoded_docs = t.texts_to_matrix(words, mode='tfidf')
    print(encoded_docs)
    vocab_size = len(t.word_index)+1
    print("Vocab Size: " + str(vocab_size))
    x = encoded_docs

    #CLUSTERING - NEW 
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(x)
    y_kmeans = kmeans.predict(x)
    

    # plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=15, cmap='viridis')
    # plt.show()
    centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, alpha=0.5)
    # plt.show()
    print(y_kmeans)


    #OLD 
    # sequences = t.texts_to_sequences(words)
    # x = pad_sequences(sequences, maxlen= max_length)
    # print(x)
    
    # #Test/Train Splitting
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state = 40) 
    # print("\nLength of XTest: " + str(len(x_test))) #Test is 40% total

    # max_words = vocab_size
    # # maxlen = max_length
    # model = Sequential()
    # model.add(Embedding(max_words, 50, input_length=x.shape[1])) #input
    # model.add(LSTM(40, activation="sigmoid", dropout=0.1, recurrent_dropout=0.1))   #hidden
    # model.add(Dense(20, activation='sigmoid'))   
    # model.add(Dense(3, activation='softmax'))                    #output
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())

    # # # #Train
    # history = model.fit(x_train, y_train, epochs=30, verbose = 1, batch_size=1, validation_split=0.1)
    
    # #LSTM predection on test dataset
    # y_pred = model.predict(x_test)
    # np.around(y_pred) == y_test
    # y_test = np.array(y_test)
    # print('predicted:', np.around(y_pred[1]))
    # print('true:', y_test[1])

    # # print('Predection')
    # y_pred = np.around(y_pred,0)
    # # print(y_pred)
    # # print('Test')
    # # print(y_test)
    # # # print("y_train:") 
    # # # print(y_train)
    # # # print("y_test:") 
    # # # print(y_test)
    # # # print(type(y_test))


    # # # #LSTM predection on test dataset
    # # # y_pred = model.predict(x_test)
    # # # np.around(y_pred) == y_pred
    # # # y_pred = np.array(y_pred)
    # # # print(y_pred)
    # # # # print('predicted:', np.around(y_pred[1]))
    # # # print(y_test)

    # # #Evaluate the Model
    # # from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    # # print("\nConfusion Matrix: ")
    # # print(confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1)))
    # # print(classification_report(y_test.argmax(axis=1),y_pred.argmax(axis=1)))
    # # print(accuracy_score(y_test.argmax(axis=1),y_pred.argmax(axis=1)))



    # # prediction = '/Users/vasubhog/Desktop/Fall Semester 2019/Requirments of Enginnering/ReqsTracing/predict.txt'

    # # # yt = list(y_test)
    # # # yp = list(y_pred)
    # # # print(yt)
    # # # print("HELOOOOOOO")
    # # # print(yp)
    # # # newTest = yt + yp
    # # # print(newTest)
    # # # f = open(prediction,'w')
    # # # for x in newTest:
    # # #     f.write("{}\n".format(str(x)))



