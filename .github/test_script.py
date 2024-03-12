with open('.github/workflows/creditmodel.sav', 'rb') as f:
    pickeled_model = pickle.load(f)

import numpy as np