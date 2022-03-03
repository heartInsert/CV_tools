import numpy as np
from sklearn.model_selection import KFold

X = np.random.randint(1, 100, 20).reshape((10, 2))

kf = KFold(n_splits=5, shuffle=True, random_state=2022)

for X_train, X_test in kf.split(X):
    print(X_train, X_test)

