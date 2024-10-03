import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

train = pd.read_csv('/workspaces/codespaces-jupyter/cars_hyundai.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(train.info())

# Display the first few rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(train.head())

# Display summary statistics of the dataset
print("\nSummary statistics of the dataset:")
print(train.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(train.isnull().sum())

# Replace categorical data in 'MaintenanceType' with numerical values
train['MaintenanceType'] = train['MaintenanceType'].astype('category').cat.codes

# Display basic information about the dataset
print("Dataset Information:")
print(train.info())

# Display the first few rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(train.head())

# Display summary statistics of the dataset
print("\nSummary statistics of the dataset:")
print(train.describe())

training, test = train_test_split(train, test_size=0.2, random_state=42)

# Assuming 'price' is the target variable and the rest are features
X_train = training.drop(columns=['AnomalyIndication'])
y_train = training['AnomalyIndication']
X_test = test.drop(columns=['AnomalyIndication'])
y_test = test['AnomalyIndication']

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Apply a threshold to convert probabilities to binary outcomes
y_pred_binary = (y_pred > 0.3).astype(int)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# Show the plot
plt.show()
plt.savefig('cm.png')
plt.close()
