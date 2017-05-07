import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing # cross_validation we don't need this because this is unsupervised
from sklearn.cluster import MeanShift
import pandas as pd

style.use('ggplot')

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))        

    return df


if __name__ == '__main__':
    
    df = pd.read_excel('titanic.xls')

    original_df = pd.DataFrame.copy(df)
    
    df.drop(['body','name'], 1, inplace=True)
    df.convert_objects(convert_numeric=True)
    df.fillna(0, inplace=True)

    df = handle_non_numerical_data(df)

    df.drop(['ticket', 'home.dest'], 1, inplace=True) #just playing arround if any other feature has impact 
    
    X = np.array(df.drop(['survived'], 1).astype(float))
    X = preprocessing.scale(X)      # improves accuracy
    y = np.array(df['survived'])

    clf = MeanShift()
    clf.fit(X)

    labels = clf.labels_
    cluster_centers = clf.cluster_centers_

    original_df['cluster_group'] = np.nan

    for i in range(len(X)):
        original_df['cluster_group'].iloc[i] = labels[i]

    n_clusters_ = len(np.unique(labels))

    # key is cluster group, value is survival rate
    survival_rates = {}

    for i in range(n_clusters_):
        # basicaly, get a new DF with the condition [ ... ] is met
        temp_df = original_df[ (original_df['cluster_group'] == float(i)) ]
        survival_cluster = temp_df[ (temp_df['survived'] == 1) ]
        survival_rate = len(survival_cluster)/len(temp_df)
        survival_rates[i] = survival_rate
    
    print(survival_rates)

##    correct = 0
##    for i in range(len(X)):
##        predict_me = np.array(X[i].astype(float))
##        predict_me = predict_me.reshape(-1, len(predict_me))
##        prediction = clf.predict(predict_me)
##        if prediction[0] == y[i]:
##            correct += 1
##
##    # because the cluster are asigned arbitrariry
##    # with, say, 80% correct rate, we'll be getting values 0.8 or 0.2 at random
##    print(correct/len(X))






    
