import numpy as np

def predict_patient(data, scaler, model):
    data_scaled = scaler.transform([data])
    prob = model.predict_proba(data_scaled)[0][1]
    label = int(prob > 0.5)

    return label, prob