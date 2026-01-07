from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm


def run_polynomial_regression(X, y, degree=2):
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(X)

    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)

    # OLS
    X_poly_sm = sm.add_constant(X_poly)
    model = sm.OLS(y, X_poly_sm)
    ols_summary = model.fit().summary()

    r2 = r2_score(y, lin_reg.predict(X_poly))

    return r2, ols_summary
