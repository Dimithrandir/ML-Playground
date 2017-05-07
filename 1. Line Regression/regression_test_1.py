import pandas
import quandl
import math
import random
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style


if __name__  == '__main__':


    x = [[0,0]]
    y = []

    for i in range(49):
        x.append([x[-1][0]+random.randint(1,5),x[-1][1]+random.randint(1,10)])
    
    y = range(50)
    

##    plt.axis([0, 50, 0, 50])
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('Horsing around')
    plt.plot(np.array(x)[:,0], np.array(x)[:,1], 'k+')
    plt.show()

    X = np.array(x)
    Y = np.array(y)
    X = preprocessing.scale(X)

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.5)

    clf = LinearRegression()
    clf.fit(X_train, Y_train)
  
    R_2 = clf.score(X_test, Y_test)


    print(R_2)


   
    
##    quandl.ApiConfig.api_key = 'YXSMoYsc5xAHeJ22kNYy'
##
##    df = []
##    try:
##        df = quandl.get('WIKI/GOOGL')
##    except Exception as e:
##        print(str(e))
##        
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

##    X = numpy.array(df.drop(['label'], 1))
##    X = preprocessing.scale(X)
##    X_lately = X[-forecast_out:]
##    X = X[:-forecast_out]
##    df.dropna(inplace=True)
##    y = numpy.array(df['label'])
##    y = numpy.array(df['label'])
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
##    df['Forecast'] = numpy.nan
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
##        df.loc[next_date] = [numpy.nan for _ in range(len(df.columns)-1)] + [i]
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
##    X = numpy.array(df.drop(['label'],1))
##    #label
##    y = numpy.array(df['label'])
##    #normalize
##    X = preprocessing.scale(X)
##    y = numpy.array(df['label'])
##
##    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
##
##    clf = LinearRegression(n_jobs = 10) #svm.SVR(kernel = 'poly') 
##    clf.fit(X_train, y_train)
##    #R^2
##    accuracy = clf.score(X_test, y_test)
##    print(accuracy)

    












    
