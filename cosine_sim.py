import numpy as np
import pandas as pd
import os
import string
from pprint import pprint


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords


# #supress warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#START PROGRAM

if __name__ == "__main__":
    #TESTING FILES 
    #run1 - 3 NFR and 80 FR
    # inputf = 'input files/1.requirements-3nfr-80fr.txt'

    #run2 - 3 NFR and 100 FR
    # inputf = 'input files/2.requirements-3nfr-100fr_Nov7.txt'

    #run3 - 4 NFR and 100 FR
    inputf = 'input files/3.requirements-4nfr-100fr_Nov7.txt'

    #Initialize
    NFR = []
    FR = []
    count = 0

    #Create two lists to split NFR and FR
    with open(inputf,'r') as input:
        for x in input:
            if x.rstrip():
                if 'NFR' in x:
                    NFR.append(x)
                else:
                    FR.append(x)

    NFRd = dict(s.split(':',1) for s in NFR)
    FRd = dict(s.split(':',1) for s in FR)
    
    result = []
    for fr in FRd.values():
        FRxNFRs =[]
        for nfr in NFRd.values():
            x_list = word_tokenize(fr.lower())
            y_list = word_tokenize(nfr.lower())

            stop = stopwords.words('english') + list(string.punctuation)

            x_list = {i for i in x_list if i not in stop}
            y_list = {i for i in y_list if i not in stop}

            l1 =[]
            l2 =[]
            
            # form a set containing keywords of both strings  
            rvector = x_list.union(y_list)  
            # print(rvector)
            for w in rvector: 
                if w in x_list: l1.append(1) # create a vector 
                else: l1.append(0) 
                if w in y_list: l2.append(1) 
                else: l2.append(0) 
            c = 0
            # print(rvector)
            # print(x_list)
            # print(l1)

            # cosine formula  
            for i in range(len(rvector)): 
                    c+= l1[i]*l2[i] 
            cosine = c / float((sum(l1)*sum(l2))**0.5)

            if cosine >= .20:
                FRxNFRs.append(1)
                count = count + 1
            else:
                FRxNFRs.append(0)

        result.append(FRxNFRs)
        # print(result)
    print(count)

    #Change Run Numbers 
    with open('VasuBhogRun3.txt','w') as file:
        for x in range(len(result)):
            if len(result[x]) == 3:
                file.write("FR{},{},{},{}\n".format(x+1,result[x][0],result[x][1],result[x][2]))
            else:
                file.write("FR{},{},{},{},{}\n".format(x+1,result[x][0],result[x][1],result[x][2],result[x][3]))


