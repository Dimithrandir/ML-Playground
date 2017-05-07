import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm

if __name__ == '__main__':

    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?', inplace=True)
    df.drop(['id'], 1, inplace=True)

    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    X_train, y_train, X_test, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    clf = svm.SVC()     # so, we just went through all the parameters for this, as described in sklearn doc 
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    
