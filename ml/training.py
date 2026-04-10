import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import joblib

def load_and_train():
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols] = df[cols].replace(0, np.nan)

    imputer = SimpleImputer(strategy='median')
    df[cols] = imputer.fit_transform(df[cols])

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        acc = (m.predict(X_test) == y_test).mean()
        results[name] = acc

    rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)

    joblib.dump(rf_model, "assets/model.pkl")
    joblib.dump(scaler, "assets/scaler.pkl")

    return df, X_scaled, y, scaler, rf_model, results