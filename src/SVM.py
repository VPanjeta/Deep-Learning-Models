from numpy import array, linspace
from matplotlib import pyplot
from matplotlib import style
from sklearn.svm import SVC
style.use("seaborn-deep")
x = [5, 1, 8, 1.5, 1, 9, 2, 4]
y = [8, 2, 8, 1.8, 0.6, 11, 4, 6]
pyplot.scatter(x,y)
pyplot.show()
X = array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11],
              [2,11],
              [4,6]])
y = [0,1,0,1,0,1,0,0]

clf =SVC(kernel='linear', C = 1.0)
clf.fit(X,y)
print(clf.predict([0.58,0.76]))
print(clf.predict([10.58,10.76]))
w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = pyplot.plot(xx, yy, 'k-', label="non weighted div")

pyplot.scatter(X[:, 0], X[:, 1], c = y)
pyplot.legend()
pyplot.show()