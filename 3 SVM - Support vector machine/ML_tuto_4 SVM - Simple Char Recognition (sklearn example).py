import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

if __name__ == '__main__':

    clf = svm.SVC()

    X,y = digits.data[:-10],digits.target[:-10]

    clf.fit(X,y)

    prediction = clf.predict(digits.data[-1])
    print(prediction)
    plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
