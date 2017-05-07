import numpy as np
from sklearn import preprocessing, cross_validation, neighbors 
import pandas as pd

if __name__ == '__main__':
    
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)

    X = np.array(df.drop(['class'], 1))        #features
    y = np.array(df['class'])        #values/class/whateva

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)   #take 20% of the data, shuffle it and return training and testing data sets
    
    clf = neighbors.KNeighborsClassifier()                  #create K nearest neighbors classifier
    clf.fit(X_train, y_train)                               #fit our training data into it

    accuracy = clf.score(X_test, y_test)                    #get the accuracy on the trained cassifier with our testing data
    print(accuracy)

    #let's make a prediction
    example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,1,1,5,1,3,3,2,1]])
    example_measures = example_measures.reshape(len(example_measures),-1)       #some normalization thing

    prediction = clf.predict(example_measures)
    print(prediction)
