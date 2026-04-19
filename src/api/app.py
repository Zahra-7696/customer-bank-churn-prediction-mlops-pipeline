import os
import joblib
import pandas as pd

from flask import Flask, request, render_template

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

model = joblib.load(os.path.join(MODELS_DIR, "final_model.joblib"))
training_columns = joblib.load(os.path.join(MODELS_DIR, "training_columns.joblib"))

print("Loaded local model successfully")

app = Flask(__name__, template_folder=TEMPLATES_DIR)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None)


@app.route("/predict-form", methods=["POST"])
def predict_form():
    try:
        user_input = {
            "CreditScore": int(request.form["CreditScore"]),
            "Geography": request.form["Geography"],
            "Gender": request.form["Gender"],
            "Age": int(request.form["Age"]),
            "Tenure": int(request.form["Tenure"]),
            "Balance": float(request.form["Balance"]),
            "NumOfProducts": int(request.form["NumOfProducts"]),
            "HasCrCard": int(request.form["HasCrCard"]),
            "IsActiveMember": int(request.form["IsActiveMember"]),
            "EstimatedSalary": float(request.form["EstimatedSalary"]),
        }

        input_df = pd.DataFrame([user_input])

        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

        prediction = model.predict(input_encoded)[0]

        if prediction == 1:
            result = "Prediction: This customer is likely to churn."
        else:
            result = "Prediction: This customer is likely to stay."

        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)