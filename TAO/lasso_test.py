from sklearn import linear_model
clf = linear_model.Lasso(alpha=.00001, fit_intercept=True)
import numpy as np

l1 = np.array(list(range(100)))
l2 = np.array(list(range(100)))

np.random.shuffle( l1 )
np.random.shuffle( l2 )

clf.fit(np.array(zip(l1,l2)), np.array(list(map( sum ,zip(l1,l2)))))
print(clf.coef_)
print(clf.intercept_)
print(clf.predict([[0,0]]))
print(clf.predict([[15,16]]))

