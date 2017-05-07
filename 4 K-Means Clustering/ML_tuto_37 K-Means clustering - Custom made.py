import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        # just take the first k feature sets and make them starting centroids
        # you gotta start somewhere amirite?
        for i in range(self.k):
            self.centroids[i] = data[i]

        # now iterate
        for i in range(self.max_iter):

            # key is just a number, value is a list of feature sets
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []    

            # now we're goind through the data 
            for featureset in data:
                # calculate the distanceses to each (current) centroid
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                # our class. is the one with minimal distance
                classification = distances.index(min(distances))
                # add this feature set to classifications dict.
                self.classifications[classification].append(featureset)

            # keep the old centroids for now 
            prev_centroids = dict(self.centroids)

            # the new centroid for each class should be the average of all the feature sets classified within it
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            # now compare all the old and new centroids
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]

                # if the distance between any of the old and new centroid pairs is larger than our predefined tolerance
                # it means we're not done optimizing
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    optimized = False

            # if we're done optimizing, then break and bye-bye
            if optimized:
                break
                
    def predict(self, data):
        # just calculate the distance between all (now optimized) centroids and give me the minimal
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))

        return classification


if __name__ == '__main__':
    
    colors = 10*['g','r','c','b','k']

    X = np.array([[1,2],
                  [1.5,1.8],
                  [5,8],
                  [8,8],
                  [1,0.6],
                  [9,11]])

    clf = K_Means()
    clf.fit(X)

    for centroid in clf.centroids:
        plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker='x', color='k', s=150, linewidth=5)

    for classification in clf.classifications:
        color = colors[classification]

        for featureset in clf.classifications[classification]:
            plt.scatter(featureset[0], featureset[1], marker='o', color=color) 

    unk = np.array([[1,3],
                  [8,9],
                  [0,3],
                  [5,6],
                  [6,4]])

    for u in unk:
        classification = clf.predict(u)
        plt.scatter(u[0], u[1], marker='*', color=colors[classification], s=100)
        
    plt.show()















    
