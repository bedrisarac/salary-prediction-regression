from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm


def run_linear_regression(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    # OLS (gerçek y ile)
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm)
    ols_summary = model.fit().summary()

    r2 = r2_score(y, lin_reg.predict(X))

    return r2, ols_summary
