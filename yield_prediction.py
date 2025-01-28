# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Simulate semiconductor process data
# Features: Process parameters (e.g., temperature, pressure, time)
# Target: Yield percentage
np.random.seed(42)
num_samples = 1000
temperature = np.random.uniform(100, 200, num_samples)  # Temperature in Celsius
pressure = np.random.uniform(1, 10, num_samples)       # Pressure in atm
time = np.random.uniform(10, 60, num_samples)          # Time in minutes

# Simulate yield based on process parameters (example formula)
yield_percent = 80 + 0.5 * temperature - 1.2 * pressure + 0.3 * time + np.random.normal(0, 5, num_samples)
yield_percent = np.clip(yield_percent, 0, 100)  # Ensure yield is between 0% and 100%

# Create a DataFrame
data = pd.DataFrame({
    'Temperature': temperature,
    'Pressure': pressure,
    'Time': time,
    'Yield': yield_percent
})

# Display the first 5 rows of the dataset
print("Sample Data:")
print(data.head())

# Split data into features (X) and target (y)
X = data[['Temperature', 'Pressure', 'Time']]
y = data['Yield']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Visualize actual vs predicted yield
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Perfect prediction line
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs Predicted Yield')
plt.show()