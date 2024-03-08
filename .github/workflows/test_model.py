import pickle
model_path = 'c:\\Users\\m.naja\\P7\\.github\\workflows\\creditmodel.sav'
credit_model = pickle.load(open(model_path, 'rb'))

import unittest

class Test_TestIncrementDecrement(unittest.TestCase):
    def test_increment(self):
        self.assertEqual(credit_model(3), 4)
    
    # This test is designed to fail for demonstration purposes.
    def test_decrement(self):
        self.assertEqual(credit_model(3), 4)

if __name__ == '__main__':
    unittest.main()