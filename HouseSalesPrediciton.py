
# STEP 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# STEP 2: Load Dataset
url = 'C:/Users/mouni/Downloads/BostonHousing.csv'
df = pd.read_csv(url)

# STEP 3: Explore Data
print("First 5 rows of data:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Stats:")
print(df.describe())

# STEP 4: Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# STEP 5: Prepare Data
X = df.drop('medv', axis=1)  # 'medv' is the target column
y = df['medv']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 6: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 7: Predict & Evaluate
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation:")
print("Mean Squared Error (MSE):", round(mse, 2))
print("R² Score:", round(r2, 2))

# STEP 8: Plot Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='green', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Fit')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)
plt.show()
