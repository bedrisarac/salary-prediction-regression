from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def run_random_forest(X, y):
    rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
    rf_reg.fit(X, y)

    r2 = r2_score(y, rf_reg.predict(X))

    return r2
