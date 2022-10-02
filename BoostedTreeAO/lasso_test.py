from sklearn import linear_model
clf = linear_model.Lasso(alpha=.00001, fit_intercept=True)
import numpy as np

l1 = np.array(list(range(100)))
l2 = np.array(list(range(100)))
np.random.shuffle( l1 )
np.random.shuffle( l2 )

y = np.array(list(map( lambda x:(1+x[0]+x[1], 2+x[0]+x[0]), zip(l1,l2))))

clf.fit(np.array(list(zip(l1,l2))), y)
print(clf.coef_)
print(clf.intercept_)
print(clf.predict([[0,0]]))
print(clf.predict([[10,10]]))

