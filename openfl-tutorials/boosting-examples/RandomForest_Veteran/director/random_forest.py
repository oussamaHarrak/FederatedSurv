from sksurv.ensemble import RandomSurvivalForest

class MyRandomSurvivalForest(RandomSurvivalForest):
    def __init__(self):
        random_state = 42
        super(MyRandomSurvivalForest, self).__init__(n_estimators=1000,min_samples_split=10,
                           min_samples_leaf=15,
                           n_jobs=-1,
                           random_state=random_state)
