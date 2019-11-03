import numpy as np
import pandas as pd
import os
import csv

#keras is a backend API for tensorflow
# import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Embedding
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

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

    df.to_csv('Combined.csv')
    # df.info()
    df.head()

    y = df[['NFR1', 'NFR2', 'NFR3']]
    # vectorizer = CountVectorizer()
    # text = df['Message']
    # vectorizer.fit(text)
    # print(vectorizer.vocabulary_)
    words = df['Message']
    max_length = max([len(s.split())for s in words])

    #Tokenize words
    t = Tokenizer(lower=True)
    t.fit_on_texts(words)
    vocab_size = len(t.word_index)+1
    print(vocab_size)
    sequences = t.texts_to_sequences(words)
    x = pad_sequences(sequences, maxlen= max_length)
    print(x)

    #TFIDF
    # t = Tokenizer(lower=True)                                      
    # t.fit_on_texts(words)
    # encoded_docs = t.texts_to_matrix(words, mode='tfidf')
    # print(encoded_docs)
    # vocab_size = len(t.word_index)+1
    # print("Vocab Size: " + str(vocab_size))
    # x = encoded_docs

    #Machine Learning Model - Sklearn

    #Test/Train Splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 40) 
    print("\nLength of XTest: " + str(len(x_test))) #Test is 40% total

    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # x_train = sc.fit_transform(x_train)
    # x_test = sc.transform(x_test)

    max_words = vocab_size
    maxlen = max_length
    model = Sequential()
    model.add(Embedding(max_words, 50, input_length=x.shape[1])) #input
    model.add(LSTM(40, activation="sigmoid", dropout=0.1, recurrent_dropout=0.1))                  #hidden
    model.add(Dense(20, activation='relu'))   
    model.add(Dense(3, activation='softmax'))                    #output
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    #Train
    history = model.fit(x_train, y_train, epochs=50, verbose = 2, batch_size=1, validation_split=0.1)
    # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00001)]

    #LSTM predection on test dataset
    pred = model.predict(x_test)
    np.around(pred) == y_test
    y_test = np.array(y_test)
    print('predicted:', np.around(pred[1]))
    print('true:', y_test[1])

 #Evaluate the Model
    # from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    # print("\nConfusion Matrix: ")
    # print(confusion_matrix(y_test.argmax(axis=1),pred.argmax(axis=1)))
    # print(classification_report(y_test.argmax(axis=1),pred.argmax(axis=1)))
    # print(accuracy_score(y_test.argmax(axis=1),pred.argmax(axis=1)))

    #Tests entire model
    y_pred = model.predict(x)
    y_pred = np.around(y_pred)
    y_pred = y_pred.astype(int)
    # print(y_pred)
    # print(len(y_pred))
    with open('result.txt','w') as result:
        for x in range(80):
            result.write("FR{},{},{},{}\n".format(x+1,int(y_pred[x][0]),y_pred[x][1],y_pred[x][2]))

