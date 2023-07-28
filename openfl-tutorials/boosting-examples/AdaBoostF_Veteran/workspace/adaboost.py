import numpy as np
from sksurv.functions import StepFunction

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
        self.n_estimators_ += 1
    def get(self, index):
        return self.estimators_[index]

    def replace(self, weak_learner, coeff):
        self.estimators_ = [weak_learner]
        self.estimator_weights_ = np.array([coeff])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        weighted_preds = np.zeros((self.n_estimators_ ,  np.shape(X)[0])) #Store the weighted predictions for each estimator
        for i, clf in enumerate(self.estimators_):
            pred = clf.predict(X)
            weighted_preds[i,:] = self.estimator_weights_[i] * pred
        y_pred = np.sum(weighted_preds ,axis = 0 ) #Calculate the weighted sum of the predictions 
        y_pred = y_pred / (self.n_estimators_)
        return y_pred  
    
    def predict_surv_function(self, X: np.ndarray) -> np.ndarray:
        preds = np.empty((self.n_estimators_ , X.shape[0]) , dtype = object)
        for i, clf in enumerate(self.estimators_):
            pred = clf.predict_survival_function(X)
            for j in range(X.shape[0]):
                new_values = pred[j].y * self.estimator_weights_[i]
                # Create a new StepFunction with the multiplied values
                new_step_function = StepFunction(pred[j].x, new_values)
                preds[i,j] = new_step_function
        survival_prob = np.zeros((X.shape[0],len(preds[0,0].x)) , dtype = object)  
        for j in range(X.shape[0]):
            for i in range(self.n_estimators_):
                survival_prob[j,:] += preds[i,j].y
            
        survivals = np.empty(X.shape[0] , dtype = object)
        for i in range(X.shape[0]):
            new_value = survival_prob[i] / self.n_estimators_
            survivals[i] = StepFunction(pred[X.shape[0]-1].x, new_value)
        return survivals
    