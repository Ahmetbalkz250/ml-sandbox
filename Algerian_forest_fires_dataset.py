import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv(r"Algerian_forest_fires_dataset.csv")
df =df.dropna().reset_index(drop=True)
df.columns = df.columns.str.strip()
df['Classes'] = df['Classes'].str.strip().map({'fire': 1, 'not fire': 0})
df = df[df['day'] != 'day']
columns_to_numeric = df.columns[:-1]
df[columns_to_numeric] = df[columns_to_numeric].astype(float)
X = df.drop("Classes", axis=1)
y = df["Classes"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)

log_model = LogisticRegression()

log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

new_Data_Input = pd.DataFrame({
    'day': [15, 20],
    'month': [8, 1],             
    'year': [2012, 2012],
    'Temperature': [38.0, 12.0],
    'RH': [30.0, 85.0],          
    'Ws': [18.0, 20.0],          
    'Rain': [0.0, 12.5],        
    'FFMC': [92.5, 40.2],        
    'DMC': [50.1, 5.1],
    'DC': [120.5, 15.2],
    'ISI': [12.0, 0.4],
    'BUI': [45.5, 4.5],
    'FWI': [25.5, 0.1]          
})

X_new = std_scaler.transform(new_Data_Input)

y_new_pred = log_model.predict(X_new)

print("\n--- RESULTS OF THE PREDICTION ---")
print(f"1. Scenario (August Inferno): {'A fire breaks out; there might not be an overfitting problem..(1) 🔥' if y_new_pred[0] == 1 else 'No problem, It could be overfitting. (0) 🌲'}")
print(f"2. Scenario (January Cold): {'A fire breaks out; there might not be an overfitting problem..(1) 🔥' if y_new_pred[1] == 1 else 'No problem, It could be overfitting. (0) 🌲'}")