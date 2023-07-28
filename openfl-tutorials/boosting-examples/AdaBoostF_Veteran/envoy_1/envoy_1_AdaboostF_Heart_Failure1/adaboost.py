import numpy as np


class AdaBoostF:
    def __init__(self, base_estimator):
        self.estimators_ = [base_estimator]
        self.n_estimators_ = 1
        self.estimator_weights_ = [1]

    def get_estimators(self):
        return self.estimators_

    def add(self, weak_learner, coeff):
        self.estimators_.append(weak_learner)
        self.estimator_weights_ = np.append(self.estimator_weights_, coeff)

    def get(self, index):
        return self.estimators_[index]

    def replace(self, weak_learner, coeff):
        self.estimators_ = [weak_learner]
        self.estimator_weights_ = np.array([coeff])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.zeros((np.shape(X)[0]))
        for i, clf in enumerate(self.estimators_):
            pred = clf.predict(X)
            for j in range(len(pred)): #for j,c in enumerate(pred)
                y_pred[j] += self.estimator_weights_[i]
        return y_pred    