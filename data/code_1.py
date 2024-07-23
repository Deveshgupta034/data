import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
data = pd.read_csv('CSUSHPISA.csv')  # Replace with your data source

# Data Cleaning
data.fillna(method='ffill', inplace=True)

# Ensure only numeric columns for correlation
numeric_data = data.select_dtypes(include=[np.number])

# EDA
sns.heatmap(numeric_data.corr(), annot=True)
plt.show()

# Feature Selection
X = data[['GDP', 'CPI', 'Interest_Rate', 'Population']]  # Example features
y = data['Home_Price_Index']  # Target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = Lasso(alpha=0.0576)  # Example alpha from previous findings
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))
