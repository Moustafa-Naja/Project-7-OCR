with open('.github/workflows/creditmodel.sav', 'rb') as f:
    pickeled_model = pickle.load(f)

import numpy as np

# Load the test data
X_test = np.load('.github/workflows/final_df_5.csv')
y_test = np.load('.github/workflows/final_df_5.csv')
