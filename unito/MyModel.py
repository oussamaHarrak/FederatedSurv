from sklearn.ensemble import RandomForestClassifier


class MyRandomForestClassifier(RandomForestClassifier):
    def __init__(self):
        super(MyRandomForestClassifier, self).__init__(n_estimators=2)
