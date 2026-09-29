# Algerian Forest Fires Risk Prediction 🔥

## Project Overview
This project aims to predict the risk of forest fires based on meteorological conditions using the Algerian Forest Fires dataset.

## Data Preprocessing
To ensure model reliability and prevent data leakage, the following preprocessing steps were executed:
* **Data Cleaning:** Handled and removed missing `NaN` values and eliminated hidden string artifacts scattered across the dataset rows.
* **Categorical Encoding:** Transformed categorical text labels into binary numerical values using the `.map()` function (e.g., mapping 'fire' to 1, 'not fire' to 0).
* **Standardization:** Applied `StandardScaler` to bring distinct meteorological metrics (Temperature, Relative Humidity, Wind Speed, etc.) onto a uniform scale.

## Model & Performance
The classification task was powered by a **Logistic Regression** model.
* **Test Score:** The model achieved a highly robust **95% Accuracy** on the unseen test dataset.
## Note
This repository was created while I was devoloping my skills in the field of Machine Learning...
