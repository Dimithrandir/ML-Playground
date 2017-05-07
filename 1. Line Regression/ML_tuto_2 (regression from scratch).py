from statistics import mean
import math, random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

def find_m(xs, yx):
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) /
           ((mean(xs)*mean(xs)) - mean(xs*xs)) )
    return m

def find_b(xs, ys, m):
    b = mean(ys) - (m * mean(xs))
    return b

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)

def find_r_squared(ys_orig, ys_line):           #coefficient of determination
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

def create_dataset(n, variance, step=2, correlation=False):             #for testing (correlation = 'pos' | 'neg'
    val = 1
    ys = []
    for i in range(n):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

if __name__  == '__main__':
    
##    xs = np.array([1,2,3,4,5,6], dtype=np.float64)
##    ys = np.array([5,4,6,5,6,7], dtype=np.float64)

    xs, ys = create_dataset(40, 40, correlation='neg')
    
    m = find_m(xs, ys)
    b = find_b(xs, ys, m)

    regression_line = [(m*x)+b for x in xs] #range(10)]

    r_2 = find_r_squared(ys, regression_line)
    
    print('m = ', m, '\nb = ', b, '\nr^2 = ', r_2)

    plt.scatter(xs, ys)
    plt.plot(xs, regression_line)
    plt.show()




    
