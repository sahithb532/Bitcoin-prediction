import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# Load data
df = pd.read_csv("bitcoin.csv")

# Drop Date column
df.drop(['Date'], axis=1, inplace=True)

# Set prediction days
predictionDays = 30

# Shift 'Price' column to create labels
df['Prediction'] = df[['Price']].shift(-predictionDays)

# Remove NaN rows for training
df.dropna(inplace=True)

# Create feature and target arrays
x = np.array(df.drop(['Prediction'], axis=1))
y = np.array(df['Prediction'])

# Split data into train/test sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Select the last 'predictionDays' rows for future prediction
predictionDays_array = np.array(df.drop(['Prediction'], axis=1))[-predictionDays:]

# Train Support Vector Machine model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma='scale')
svr_rbf.fit(xtrain, ytrain)

# Evaluate model
svr_rbf_confidence = svr_rbf.score(xtest, ytest)
print('SVR_RBF accuracy:', svr_rbf_confidence)

# Predict test set
svm_prediction = svr_rbf.predict(xtest)
print("Predictions on test set:", svm_prediction)
print("Actual test values:", ytest)

# Predict future 30 days
future_predictions = svr_rbf.predict(predictionDays_array)
print("Predictions for next 30 days:", future_predictions)
