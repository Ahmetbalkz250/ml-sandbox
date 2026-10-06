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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv(r"4-Algerian_forest_fires_dataset.csv")
df =df.dropna().reset_index(drop=True)
df.columns = df.columns.str.strip()
df['Classes'] = df['Classes'].str.strip().map({'fire': 1, 'not fire': 0})
df = df[df['day'] != 'day']
columns_to_numeric = df.columns[:-1]
df[columns_to_numeric] = df[columns_to_numeric].astype(float)
X = df.drop("Classes", axis=1)
y = df["Classes"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline = make_pipeline(StandardScaler(), LogisticRegression())
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
new_Data_Input = pd.DataFrame({
    'day': [15, 20], 'month': [8, 1], 'year': [2012, 2012],
    'Temperature': [38.0, 12.0], 'RH': [30.0, 85.0], 'Ws': [18.0, 20.0],          
    'Rain': [0.0, 12.5], 'FFMC': [92.5, 40.2], 'DMC': [50.1, 5.1],
    'DC': [120.5, 15.2], 'ISI': [12.0, 0.4], 'BUI': [45.5, 4.5], 'FWI': [25.5, 0.1]          
})
y_new_pred = model_pipeline.predict(new_Data_Input)

print("\n--- RESULTS OF THE PREDICTION ---")
print(f"1. Scenario (August Inferno): {'A fire breaks out; there might not be an overfitting problem..(1) 🔥' if y_new_pred[0] == 1 else 'No problem, It could be overfitting. (0) 🌲'}")
print(f"2. Scenario (January Cold): {'A fire breaks out; there might not be an overfitting problem..(1) 🔥' if y_new_pred[1] == 1 else 'No problem, It could be overfitting. (0) 🌲'}")

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", 
            xticklabels=["A fire won't break out. (0)", 'A fire breaks out(1)'], 
            yticklabels=["A fire won't break out. (0)", 'A fire breaks out(1)'])
plt.title('Algerian Forest Fires - Confusion Matrix')
plt.ylabel('Real Values')
plt.xlabel('Predict')
plt.tight_layout()
plt.show()