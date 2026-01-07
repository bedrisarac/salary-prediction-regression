from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


def run_decision_tree(X, y):
    dt_reg = DecisionTreeRegressor(random_state=0)
    dt_reg.fit(X, y)

    r2 = r2_score(y, dt_reg.predict(X))

    return r2
