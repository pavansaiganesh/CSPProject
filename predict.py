import joblib 

rf = joblib.load('RandomForest.pkl') 

"""eg:  arr = [[104, 18, 30, 23.6, 60, 6, 140]]"""

def predict_type(arr):
    res = rf.predict(arr)
    print(res)
    return res[0]
    