import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("kidney_disease.csv")

# Replace empty with NaN
df.replace("", np.nan, inplace=True)

# Encoding
df['classification'] = df['classification'].map({'ckd':1, 'notckd':0})

binary_map = {'yes':1, 'no':0}
for col in ['htn','dm','cad','pe','ane']:
    df[col] = df[col].map(binary_map)

df['rbc'] = df['rbc'].map({'normal':1, 'abnormal':0})
df['pc'] = df['pc'].map({'normal':1, 'abnormal':0})
df['pcc'] = df['pcc'].map({'present':1, 'notpresent':0})
df['ba'] = df['ba'].map({'present':1, 'notpresent':0})
df['appet'] = df['appet'].map({'good':1, 'poor':0})

# Convert numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Fill missing
df.fillna(df.median(), inplace=True)

# Split
X = df.drop('classification', axis=1)
y = df['classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model saved ✅")