import pickle
import pandas as pd

with open('C:/Users/m.naja/P7/.github/workflows/creditmodel.sav', 'rb') as f:
    pickeled_model = pickle.load(f)

import numpy as np


df = pd.read_csv('C:/Users/m.naja/P7/.github/workflows/final_df_5.csv')

# Load the test data
X_test = df["EXT_SOURCE_2", "EXT_SOURCE_3", "CNT_FAM_MEMBERS", "DAYS_REGISTRATION", "AMT_REQ_CREDIT_BUREAU_HOUR"]
y_test = df['TARGET']

# Make predictions
predictions = pickeled_model.predict(X_test)

# Evaluate the model
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy}")

# Assert the model performance is as expected
assert accuracy > 0.8  
