# Salary Prediction with Regression Models

This project aims to predict employee salaries using different regression algorithms
and compare their performance.

The main objective is to analyze how various regression models behave on the same dataset
and to understand their strengths and limitations.

---

## Dataset

The dataset contains employee-related information including job title level,
seniority, performance score, and salary.

**Features used in the models:**
- UnvanSeviyesi (Title Level)
- Kidem (Seniority)
- Puan (Performance Score)

**Target variable:**
- maas (Salary)

Categorical variables such as job title were not directly used in the models.

---

## Models Implemented

The following regression models are implemented in this project:

- Linear Regression  
- Polynomial Regression (degree = 2)  
- Support Vector Regression (SVR)  
- Decision Tree Regressor  
- Random Forest Regressor  

Each model is implemented as a separate module for better modularity and readability.

---

## Evaluation Metric

Model performance is evaluated using the **R² score (Coefficient of Determination)**.

R² measures how well the independent variables explain the variance in the target variable.

---

## Visualizations

Scatter plots and regression curves were used to visualize the relationship between
title level and salary for linear and polynomial regression models.

Tree-based and SVR models were evaluated using performance metrics only, as their
structures are not suitable for direct functional visualization.

---

## Project Structure

