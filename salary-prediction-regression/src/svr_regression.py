from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score


def run_svr(X, y):
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X_scaled = sc_X.fit_transform(X)
    y_scaled = sc_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    svr_reg = SVR(kernel='rbf')
    svr_reg.fit(X_scaled, y_scaled)

    r2 = r2_score(y_scaled, svr_reg.predict(X_scaled))

    return r2
