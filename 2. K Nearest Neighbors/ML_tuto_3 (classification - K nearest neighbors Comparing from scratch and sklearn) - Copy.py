import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

def k_nearest_neightbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []

    for group in data:
        for features in data[group]:
            ##euclidean_distance = np.sqrt( np.sum( (np.array(features) - np.array(predict))**2 ) )   //the hard way for ED
            eucliedean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([eucliedean_distance, group])
            
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    print(vote_result, confidence)
    
    return vote_result
        

if __name__ == '__main__':

    df = pd.read_csv('breast-cancer-wisconsin.data')    #read the cancer data
    df.replace('?', -99999, inplace=True)               #replace the missing data, denoutet with '?'
    df.drop(['id'], 1, inplace=True)                    #drop the 'id' column
    full_data = df.astype(float).values.tolist()        #convert all values to float
    random.shuffle(full_data)                           #everyday I'm shuffling
    
    test_size = 0.2                                     
    train_set = {2:[], 4:[]}                            #dictionary key '2' is the first class, its value - list with all
    test_set = {2:[], 4:[]}                             # data vectors in that 

    train_data = full_data[:-int(test_size*len(full_data))]     #split the data into training and testing set
    test_data = full_data[-int(test_size*len(full_data)):]      

    for i in train_data:                            #fill the train_set
        train_set[i[-1]].append(i[:-1])
        
    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:          
        for data in test_set[group]:        
            vote = k_nearest_neightbors(train_set, data, k=5)   #run our function on every list element in the test_set
            if group == vote:                                   #it the result is the same with the data class,
                correct += 1                                    # it means we are correct 
            total += 1

    print('Accuracy = ', correct/total)     #accuracy is just the ratio








    

    
    
