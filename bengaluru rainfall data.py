# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 22:14:32 2025

@author: patil
"""

# Bengaluru Rainfall Prediction (2021–2024)
# Spyder Compatible Version (No 'squared' argument issue)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Step 1: Load the CSV files
df_2021 = pd.read_csv("2021 data.csv")
df_2022 = pd.read_csv("2022 data.csv")
df_2023 = pd.read_csv("2023 data.csv")
df_2024 = pd.read_csv("2024 data.csv")

# Step 2: Combine all data
df = pd.concat([df_2021, df_2022, df_2023, df_2024], ignore_index=True)

# Step 3: Keep only required columns
df = df[[
    'District/Taluk/Hobli',
    'Pre-Monsoon Actual (mm)',
    'SWM Actual (mm)',
    'NEM Actual (mm)',
    'Annual Actual (mm)'
]].dropna()

# Step 4: Define features and target
X = df[['Pre-Monsoon Actual (mm)', 'SWM Actual (mm)', 'NEM Actual (mm)']]
y = df['Annual Actual (mm)']

# Step 5: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 6: Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict on the test set
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Manual RMSE fix
r2 = r2_score(y_test, y_pred)

print("✅ Model Evaluation:")
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R² Score:", round(r2, 2))

# Step 9: Predict on new data

new_input_df = pd.DataFrame([[120, 620, 140]], columns=[
    'Pre-Monsoon Actual (mm)',
    'SWM Actual (mm)',
    'NEM Actual (mm)'
])

prediction = model.predict(new_input_df)
print("🌦 Predicted Annual Rainfall:", round(prediction[0], 2), "mm")

df.to_csv("bengaluru_rainfall_combined.csv", index=False)
print("✅ Data exported for Power BI.")


