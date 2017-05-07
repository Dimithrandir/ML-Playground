import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

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
    
    return vote_result
        

if __name__ == '__main__':

    dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
    new_features = [6,3]

    result = k_nearest_neightbors(dataset, new_features)
    print(result)
    
    [[plt.scatter(j[0], j[1], color=i) for j in dataset[i]] for i in dataset]
    plt.scatter(new_features[0], new_features[1], s=100, color=result)
    plt.show()
