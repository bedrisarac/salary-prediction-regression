# libraries
import pandas as pd
from src.linear_regression import run_linear_regression
from src.polynomial_regression import run_polynomial_regression
from src.svr_regression import run_svr
from src.decision_tree_regression import run_decision_tree
from src.random_forest_regression import run_random_forest
import matplotlib.pyplot as plt
import numpy as np

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

#  Linear Regression Graph 
plt.scatter(X, y, color="red", label="Actual Data")
plt.plot(X, run_linear_regression(X, y)[0] * 0 + run_linear_regression(X, y)[0], alpha=0)
plt.plot(X, run_linear_regression(X, y)[0] * 0 + run_linear_regression(X, y)[0], alpha=0)

plt.plot(X, run_linear_regression(X, y)[0] * 0 + run_linear_regression(X, y)[0], alpha=0)
plt.plot(X, run_linear_regression(X, y)[0] * 0 + run_linear_regression(X, y)[0], alpha=0)

# real line
lin_reg = LinearRegression()
lin_reg.fit(X, y)
plt.plot(X, lin_reg.predict(X), color="blue", label="Linear Prediction")

plt.xlabel("Unvan Seviyesi")
plt.ylabel("Maas")
plt.title("Linear Regression: Salary Prediction")
plt.legend()
plt.show()


# Polynomial Regression
print('POLYNOMIAL REGRESSION ')
poly_r2, poly_summary = run_polynomial_regression(X, y, degree=2)
print(poly_summary)
print("Polynomial R2 degeri:", poly_r2)
print()

#  Polynomial Regression Graph
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Sorting for a smoother line.
X_grid = np.sort(X.values, axis=0)
X_grid_poly = poly_reg.transform(X_grid)

plt.scatter(X, y, color="red", label="Actual Data")
plt.plot(X_grid, lin_reg2.predict(X_grid_poly), color="green", label="Polynomial Prediction")

plt.xlabel("Unvan Seviyesi")
plt.ylabel("Maas")
plt.title("Polynomial Regression (Degree 2): Salary Prediction")
plt.legend()
plt.show()

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

