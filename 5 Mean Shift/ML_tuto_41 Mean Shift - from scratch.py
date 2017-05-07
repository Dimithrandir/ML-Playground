import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

class Mean_Shift:
    def __init__(self, radius=4):       # radius is actually bandwidth.
        self.radius = radius
        
    def fit(self, data):
        centroids = {}

        # every point is a centroid on start with Mean shift 
        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            new_centroids = []
            for i in centroids:
                # feature sets in the bandwidth will be kept here
                in_bandwidth = []
                centroid = centroids[i]
                # cycle all the data, if a feature set is in the radius, add it
                for featureset in data:
                    if np.linalg.norm(featureset-centroid) < self.radius:
                        in_bandwidth.append(featureset)
                        
                new_centroid = np.average(in_bandwidth,axis=0)
                new_centroids.append(tuple(new_centroid))    # we convert this to a tuple, so we can create a set after

            # this is how we deal with convergence. After a while, many centroids will overlap
            # so we don't need duplicates, only the uniques
            uniques = sorted(list(set(new_centroids)))

            # update centroids now
            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            # if all new centroids are the same with the last ones, it means we've optimized them
            # else, break and continue with the new centroids...
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
                
            if optimized:
                break

        self.centroids = centroids
                
    def predict(self, data):
        pass


if __name__ == '__main__':
    
    colors = 10*['g','r','c','b','k']

    X = np.array([[1,2],
                  [1.5,1.8],
                  [5,8],
                  [8,8],
                  [1,0.6],
                  [9,11],
                  [8,2],
                  [10,2],
                  [9,3],])


    clf = Mean_Shift()
    clf.fit(X)

    centr = clf.centroids

    print(centr)
    
    plt.scatter(X[:,0], X[:,1])
    for c in centr: 
        plt.scatter(centr[c][0], centr[c][1], color='k', marker='*', s=150)
    plt.show()
    















    
