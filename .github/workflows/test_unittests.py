import pickle
import unittest
import numpy as np
import pandas as pd

application_test = pd.read_csv('C:\\Users\\m.naja\\Desktop\\application_test.csv')
application_test = application_test.dropna()
features = ["EXT_SOURCE_2", "EXT_SOURCE_3", "CNT_FAM_MEMBERS", "DAYS_REGISTRATION", "AMT_REQ_CREDIT_BUREAU_HOUR"]


model_path = 'c:\\Users\\m.naja\\P7\\.github\\workflows\\creditmodel.sav'
credit_model = pickle.load(open(model_path, 'rb'))

class TestKNNModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the pickled model
        with open(model_path, 'rb') as file:
            cls.model = pickle.load(file)


    def test_single_prediction(self):
        # Create a test sample with n features
        test_sample = np.array(application_test[features].iloc[0]).reshape(1,-1)
        # Make a prediction
        prediction = self.model.predict(test_sample)
        # Assert the prediction is as expected
        self.assertIn(prediction,[0, 1],"La prédiction doit être 0 ou 1")

    def test_multiple_predictions(self):
        # Create multiple test samples
        test_samples = application_test[[
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "CNT_FAM_MEMBERS",
            "DAYS_REGISTRATION",
            "AMT_REQ_CREDIT_BUREAU_HOUR"]].head(10).values
        # Make predictions
        predictions = self.model.predict(test_samples)
        # Assert the prediction is as expected
        for prediction in predictions:
            self.assertIn(prediction, [0,1],"Chaque prédiction doit être 0 ou 1")


    def test_invalid_input(self):
        # Testez le comportement du modèle avec une entrée invalide
        sample = np.array([[application_test["EXT_SOURCE_2"],application_test["EXT_SOURCE_3"]]])  # Moins de features que nécessaire
        with self.assertRaises(ValueError):
            self.model.predict(sample)

if __name__ == '__main__':
    unittest.main()