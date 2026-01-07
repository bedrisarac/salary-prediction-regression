# libraries
import pandas as pd
from src.linear_regression import run_linear_regression
from src.polynomial_regression import run_polynomial_regression
from src.svr_regression import run_svr
from src.decision_tree_regression import run_decision_tree
from src.random_forest_regression import run_random_forest

# data
df = pd.read_csv("data/data.txt")
X = df.iloc[:, -2:-1]
y = df.iloc[:, -1]

# Linear Regression
print('Linear Regression')
linear_r2, linear_summary = run_linear_regression(X, y)
print(linear_summary)
print("Linear R2 degeri:", linear_r2)
print()


# POLYNOMIAL REGRESSION
print('POLYNOMIAL REGRESSION ')
poly_r2, poly_summary = run_polynomial_regression(X, y, degree=2)
print(poly_summary)
print("Polynomial R2 degeri:", poly_r2)
print()

# SVR
print("Support Vector Regression (SVR)")
svr_r2 = run_svr(X, y)
print("SVR R2 degeri:", svr_r2)
print()

# Decision Tree Regression
print(" Decision Tree Regression ")
dt_r2 = run_decision_tree(X, y)
print("Decision Tree R2 degeri:", dt_r2)
print()

# Random Forest Regression
print(" Random Forest Regression ")
rf_r2 = run_random_forest(X, y)
print("Random Forest R2 degeri:", rf_r2)
print()

# summary
print(" R2 SCORE SUMMARY ")
print(f"Linear Regression     : {linear_r2}")
print(f"Polynomial Regression : {poly_r2}")
print(f"SVR                   : {svr_r2}")
print(f"Decision Tree         : {dt_r2}")
print(f"Random Forest         : {rf_r2}")
