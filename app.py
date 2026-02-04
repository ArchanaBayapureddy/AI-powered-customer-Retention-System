from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# ================= LOAD MODEL & SCALER =================
with open("churn.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ✅ EXACT COLUMNS USED DURING TRAINING (28 columns)
MODEL_COLUMNS = list(scaler.feature_names_in_)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            input_values = []

            # -------- READ ALL 28 INPUTS --------
            for col in MODEL_COLUMNS:
                val = float(request.form[col])
                input_values.append(val)

            # -------- CREATE DATAFRAME --------
            input_df = pd.DataFrame([input_values], columns=MODEL_COLUMNS)

            # -------- SCALE & PREDICT --------
            scaled_data = scaler.transform(input_df)
            pred = model.predict(scaled_data)[0]
            prob = model.predict_proba(scaled_data)[0][1]

            prediction = "Customer Will Churn" if pred == 1 else "Customer Will Stay"
            probability = round(prob, 2)

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        columns=MODEL_COLUMNS,
        prediction=prediction,
        probability=probability,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)

