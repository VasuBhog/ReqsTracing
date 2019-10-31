
# coding: utf-8
import numpy as np
import pandas as pd
import os

# #access to the desktop. change to the location of your file (e.g. Documents)
# os.chdir()

# #reading csv file as a dataframe. Can also read a text file with different libraries
# df = pd.read_csv('requirements.csv')

# df.head()

# y = df[['NFR1', 'NFR2', 'NFR3']]

# y

# #keras is a backend API for tensorflow
# import keras
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense, Activation, LSTM, Embedding
# from keras.callbacks import EarlyStopping

# df.FR

# tokenizer = Tokenizer(lower=True) #function intialization for tokenizer
# tokenizer.fit_on_texts(df.FR)    #tokenizing the FRs


# max_length = max([len(s.split())for s in df.FR])

# max_length


# vocab_size = len(tokenizer.word_index)+1


# vocab_size


# sequences = tokenizer.texts_to_sequences(df.FR)
# x = pad_sequences(sequences, maxlen=max_length) #adding 0's before the sentence


# x #check the padding sequences


# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=9000) #splitting the dataset into 60-40%


# len(x_test) #see the number of test samples

# max_words = vocab_size
# maxlen = max_length # same as input_length = x.shape[1]


# model = Sequential()
# model.add(Embedding(max_words, 50, input_length=x.shape[1])) #input
# model.add(LSTM(20, activation = 'sigmoid'))                  #hidden
# model.add(Dense(3, activation='softmax'))                    #output
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# history = model.fit(x_train, y_train, epochs=30, verbose = 1, batch_size=1,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])


# y_test


# pred = model.predict(x_test) #LSTM prediction on test dataset

# np.around(pred) == y_test #check no of true cases

# y_test = np.array(y_test) #making ground truth to an array using numpy


# print('predicted:', np.around(pred[1]))
# print('true:', y_test[1])


# model1 = Sequential()
# model1.add(Embedding(max_words, 50, input_length=x.shape[1]))
# model1.add(LSTM(30, activation='sigmoid'))
# model1.add(Dense(3, activation='softmax'))
# model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# history = model1.fit(x_train, y_train, epochs=30, verbose = 1, batch_size=1,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])


# pred1 = model1.predict(x_test)


# y_test1 = np.array(y_test)


# np.around(pred1) == y_test #check no. of true cases


# print('predicted:', np.around(pred1[1]))
# print('ture:', y_test1[1])


# model2 = Sequential()
# model2.add(Embedding(max_words, 50, input_length=x.shape[1]))
# model2.add(LSTM(40, activation='sigmoid'))
# model2.add(Dense(3, activation='softmax'))
# model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# history = model2.fit(x_train, y_train, epochs=30, verbose = 1, batch_size=1,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])



# pred2 = model2.predict(x_test)


# np.around(pred2) == y_test #check no. of true cases

# y_test = np.array(y_test)


# print('predicted:', np.around(pred2[1]))
# print('ture:', y_test[1])


# model2 = Sequential()
# model2.add(Embedding(max_words, 50, input_length=x.shape[1]))
# model2.add(LSTM(40, activation='relu'))
# model2.add(Dense(3, activation='softmax'))
# model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# history = model2.fit(x_train, y_train, epochs=30, verbose = 1, batch_size=1,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])


# pred2 = model2.predict(x_test)

# pred2 = np.around(pred2)

# y_test = np.array(y_test)

# print('predicted:', pred2[1])
# print('true:',y_test[1])

