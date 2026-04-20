from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])
        
        final_features = scaler.transform(final_features)
        prediction = model.predict(final_features)

        result = "CKD Detected" if prediction[0] == 1 else "No CKD"

        return render_template('index.html', prediction_text=result)

    except:
        return render_template('index.html', prediction_text="Error in input")


if __name__ == "__main__":
    app.run(debug=True)