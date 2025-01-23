import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv(r"E:\tarun jain\manufacturing\manufacturing_defect_dataset.csv")

# Data preprocessing
x = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Target

# Scaling
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# Split the data 
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

# Handling imbalanced data
smote = SMOTE()
x_train, y_train = smote.fit_resample(x_train, y_train)

# Random Forest model
rfc = RandomForestClassifier(random_state=42)
rfc.fit(x_train, y_train)

pred = rfc.predict(x_test)

accuracy = accuracy_score(pred, y_test)
report = classification_report(pred, y_test)
conf_matrix = confusion_matrix(pred, y_test)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)

import pickle
with open("random_forest_model.pkl", "wb") as model_file:
    pickle.dump(rfc, model_file)
