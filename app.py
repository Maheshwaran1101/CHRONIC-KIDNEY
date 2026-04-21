import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# -------------------------------
# Load and preprocess dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("kidney_disease.csv")

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

    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(df.median(), inplace=True)

    return df

df = load_data()

# -------------------------------
# Split data
# -------------------------------
X = df.drop('classification', axis=1)
y = df['classification']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Models
# -------------------------------
models = {
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=8),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(kernel='linear')
}

# Train models
for model in models.values():
    model.fit(X_train, y_train)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Kidney Disease Prediction App")

st.write("Enter patient details:")

# Input fields
age = st.number_input("Age", 1, 100)
bp = st.number_input("Blood Pressure")
sg = st.number_input("Specific Gravity")
al = st.number_input("Albumin")
su = st.number_input("Sugar")
rbc = st.selectbox("RBC", ["normal", "abnormal"])
pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])
ba = st.selectbox("Bacteria", ["present", "notpresent"])
bgr = st.number_input("Blood Glucose Random")
bu = st.number_input("Blood Urea")
sc = st.number_input("Serum Creatinine")
sod = st.number_input("Sodium")
pot = st.number_input("Potassium")
hemo = st.number_input("Hemoglobin")
pcv = st.number_input("Packed Cell Volume")
wc = st.number_input("White Blood Cell Count")
rc = st.number_input("Red Blood Cell Count")
htn = st.selectbox("Hypertension", ["yes", "no"])
dm = st.selectbox("Diabetes", ["yes", "no"])
cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])
appet = st.selectbox("Appetite", ["good", "poor"])
pe = st.selectbox("Pedal Edema", ["yes", "no"])
ane = st.selectbox("Anemia", ["yes", "no"])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    input_data = pd.DataFrame([[
        age, bp, sg, al, su,
        1 if rbc == "normal" else 0,
        1 if pc == "normal" else 0,
        1 if pcc == "present" else 0,
        1 if ba == "present" else 0,
        bgr, bu, sc, sod, pot, hemo,
        pcv, wc, rc,
        1 if htn == "yes" else 0,
        1 if dm == "yes" else 0,
        1 if cad == "yes" else 0,
        1 if appet == "good" else 0,
        1 if pe == "yes" else 0,
        1 if ane == "yes" else 0
    ]], columns=X.columns)

    # Scale input
    input_scaled = scaler.transform(input_data)

    st.subheader("Predictions from Models:")

    for name, model in models.items():
        pred = model.predict(input_scaled)[0]
        result = "CKD" if pred == 1 else "NOT CKD"
        st.write(f"{name}: {result}")