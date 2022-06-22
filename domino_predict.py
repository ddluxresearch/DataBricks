import pickle
import datetime
import pandas as pd

with open('DecisionTree_model.sav', 'rb') as f:
    m = pickle.load(f)

def predict(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm):
    '''
    Input:
    SepalLengthCm - float
    SepalWidthCm - float
    PetalLengthCm - float
    PetalWidthCm - float

    Output:
    Species - string
    '''
    data = [{'SepalLengthCm': float(SepalLengthCm), 'SepalWidthCm': float(SepalWidthCm), 'PetalLengthCm': float(PetalLengthCm), 'PetalWidthCm':float(PetalWidthCm)}]
    ds = pd.DataFrame(data)
    return m.predict(ds)[0]