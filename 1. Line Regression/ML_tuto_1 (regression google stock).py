import math, datetime
import quandl
import pandas
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle 

##style.use('ggplot')

if __name__  == '__main__':
    quandl.ApiConfig.api_key = 'YXSMoYsc5xAHeJ22kNYy'

    df = []
    try:
        df = quandl.get('WIKI/GOOGL')
    except Exception as e:
        print(str(e))

    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
    df['High-Low %'] = ((df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']) * 100.0
    df['Change %'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']) * 100.0
    df = df[['Adj. Close', 'High-Low %', 'Change %', 'Adj. Volume']]
    
##    print('df length = ', len(df))

    forecast = int(math.ceil(0.01*len(df)))

    print('forecast length = ', forecast)
    
    df['Label'] = df['Adj. Close'].shift(-forecast)
    df_1 = df[['Adj. Close']]

    X = np.array(df.drop(['Label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast:]
    X = X[:-forecast]
    df.dropna(inplace=True)
    y = np.array(df['Label'])

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

##    clf = LinearRegression()
##    clf.fit(X_train, y_train)
##    #saving the classifier
##    with open('linearregression.pickle', 'wb') as f:
##        pickle.dump(clf, f)
    #reading the classifier from a pickle
    pickle_in = open('linearregression.pickle', 'rb')
    clf = pickle.load(pickle_in)
    
    accuracy = clf.score(X_test, y_test)
    print('R^2 = ', accuracy)

    prediction = clf.predict(X_lately)

    df['Forecast'] = np.nan
    df['Actual'] = np.nan
    
    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in prediction:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += one_day

        if next_date.date() in df_1.index:     
            df.loc[next_date] = [np.nan for _ in range(len(df.columns)-2)] + [i] + [df_1.loc[next_date.date()]['Adj. Close']]
        else:
            df.loc[next_date] = [np.nan for _ in range(len(df.columns)-2)] + [i] + [np.nan]


    df['Adj. Close'].plot()
    df['Forecast'].plot()
    df['Actual'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

    
##    df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
##    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
##    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
##    
##    df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
##
####    НОВА КОЛОНА СО ВРЕДНОСТИ ПОМЕСТЕНИ ЗА НЕКОЛКУ ДЕНА НАНАПРЕД
##    forecast_col = 'Adj. Close'
##    df.fillna(-99999, inplace=True)
##    forecast_out = int(math.ceil(0.01*len(df)))
##    print(forecast_out)
##    df['label'] = df[forecast_col].shift(-forecast_out)
##
##    X = np.array(df.drop(['label'], 1))
##    X = preprocessing.scale(X)
##    X_lately = X[-forecast_out:]
##    X = X[:-forecast_out]
##    df.dropna(inplace=True)
##    y = np.array(df['label'])
##    y = np.array(df['label'])
##
##    
##    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
##
##    clf = LinearRegression(n_jobs = 10) #svm.SVR(kernel = 'poly') 
##    clf.fit(X_train, y_train)
##    #R^2
##    accuracy = clf.score(X_test, y_test)
##
##    forecast_set = clf.predict(X_lately)
##
##    print(forecast_set, accuracy, forecast_out)
##
##    df['Forecast'] = np.nan
##
##    last_date = df.iloc[-1].name
##
##    last_unix = last_date.timestamp()
##    one_day = 86400
##    next_unix = last_unix + one_day
##
##    for i in forecast_set:
##        next_date = datetime.datetime.fromtimestamp(next_unix)
##        next_unix += one_day
##        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
##
##    df['Adj. Close'].plot()
##    df['Forecast'].plot()
##    plt.legend(loc=4)
##    plt.xlabel('Date')
##    plt.ylabel('Price')
##    plt.show()


    
    
##  TRAINING AND TESTING
##
##    #feature
##    X = np.array(df.drop(['label'],1))
##    #label
##    y = np.array(df['label'])
##    #normalize
##    X = preprocessing.scale(X)
##    y = np.array(df['label'])
##
##    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
##
##    clf = LinearRegression(n_jobs = 10) #svm.SVR(kernel = 'poly') 
##    clf.fit(X_train, y_train)
##    #R^2
##    accuracy = clf.score(X_test, y_test)
##    print(accuracy)

    












    
