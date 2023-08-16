import unittest
import pandas as pd
from preprocess_data import load_data, preprocess_data, split_data
from train_model import train_and_evaluate
from save_load_model import save_model, load_model

class TestModel(unittest.TestCase):

    def setUp(self):
        # Load and preprocess the data
        self.data = load_data()
        self.data = preprocess_data(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.data)

    def test_data_loading(self):
        # Check if data is loaded correctly
        self.assertIsNotNone(self.data)
        self.assertTrue(isinstance(self.data, pd.DataFrame))
        self.assertNotEqual(self.data.shape[0], 0)

    def test_data_split(self):
        # Check if data is split correctly
        self.assertNotEqual(self.X_train.shape[0], 0)
        self.assertNotEqual(self.X_test.shape[0], 0)
        self.assertNotEqual(self.y_train.shape[0], 0)
        self.assertNotEqual(self.y_test.shape[0], 0)

    def test_model_training_and_evaluation(self):
        # Train the model and evaluate its accuracy
        model, accuracy = train_and_evaluate(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertIsNotNone(model)
        self.assertTrue(0 <= accuracy <= 1)  # Accuracy should be between 0 and 1

    def test_model_saving_and_loading(self):
        # Save and load the model to ensure it works correctly
        model, _ = train_and_evaluate(self.X_train, self.y_train, self.X_test, self.y_test)
        save_model(model, 'test_model.joblib')
        loaded_model = load_model('test_model.joblib')
        self.assertIsNotNone(loaded_model)

        # Ensure that predictions from the saved model and loaded model are the same
        original_predictions = model.predict(self.X_test)
        loaded_predictions = loaded_model.predict(self.X_test)
        self.assertTrue((original_predictions == loaded_predictions).all())

if __name__ == '__main__':
    unittest.main()
