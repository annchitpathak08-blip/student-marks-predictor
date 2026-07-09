import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

df = pd.read_csv("student-mat.csv", sep=";")

# Basic Exploration of the dataframe 
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

# Remove duplicates if any here are none as df was provided clean
df = df.drop_duplicates()

# One-Hot Encoding increasing the num of cols rapidly 
df = pd.get_dummies(df, drop_first=True)

# Features and Target (input is all except y)
X = df.drop("G3", axis=1)
y = df["G3"]

# Train-Test Split (training with 20 percent of total data with seed)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(coefficients.sort_values(by="Coefficient", ascending=False))

# Predict New Student
new_student = X.iloc[[0]].copy()   # Take first student's data as a template and we can do changes as we want 
new_student["studytime"] = 67
new_student["absences"] = 69
new_student["failures"] = 0

prediction = model.predict(new_student)

print("Predicted G3:", prediction[0])


plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Error Distribution
errors = y_test - y_pred
sns.histplot(errors,
             bins=10,
             kde=True,
             color="green")

plt.title("Prediction Error Distribution")
plt.show()

sns.scatterplot(x=y_test,
                y=y_pred,
                color="dodgerblue")

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
# MAE : 1.6466656197147518
# MSE : 5.656642833231225
# RMSE: 2.3783697847961376
# R² Score: 0.7241341236974019
#  scores for reference over testing data that was 0.2