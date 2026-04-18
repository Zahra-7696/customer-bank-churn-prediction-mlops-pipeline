#To run .\.venv\Scripts\python.exe src/api/app.py
#docker images
#docker run --name bank-churn-container -p 5001:5001 bank-churn-api
#or docker run -d --name bank-churn-container -p 5001:5001 bank-churn-api
#docker ps
#to stop it: docker stop bank-churn-container
#open:http://127.0.0.1:5001


# import mlflow
# import mlflow.pyfunc
# import pandas as pd

# from flask import Flask, request, jsonify
# from mlflow.tracking import MlflowClient

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# client = MlflowClient()

# model_name = "bank_churn_model"
# latest_versions = client.search_model_versions(f"name='{model_name}'")
# latest_version = max(int(mv.version) for mv in latest_versions)

# model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version}")

# print(f"Loaded model: {model_name}, version: {latest_version}")

# app = Flask(__name__)

# @app.route("/", methods=["GET"])
# def home():
#     return "Bank Churn API is running. Use POST /predict for predictions."

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()
#         df = pd.DataFrame([data])
#         prediction = model.predict(df)

#         return jsonify({
#             "prediction": prediction.tolist()
#         })
#     except Exception as e:
#         return jsonify({
#             "error": str(e)
#         }), 400

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001)


############################################################
#import mlflow
#import mlflow.pyfunc
import os
import pandas as pd
import joblib


from flask import Flask, request, render_template_string
#from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load MLflow model
# -----------------------------
#mlflow.set_tracking_uri("http://127.0.0.1:5000")
#client = MlflowClient()

#model_name = "bank_churn_model"
#latest_versions = client.search_model_versions(f"name='{model_name}'")
#latest_version = max(int(mv.version) for mv in latest_versions)

#model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version}")
#print(f"Loaded model: {model_name}, version: {latest_version}")

# -----------------------------
# Recreate training columns
# -----------------------------
df = pd.read_csv("data/raw/bank_churn_train.csv")

X = df.drop(columns=["Exited"])
y = df["Exited"]

X_encoded = pd.get_dummies(X, drop_first=True)
training_columns = X_encoded.columns.tolist()
# Split the encoded data, not raw X
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_dir = os.path.join(BASE_DIR, "models")

os.makedirs(model_dir, exist_ok=True)

joblib.dump(rf_model, "models/final_model.joblib")
joblib.dump(training_columns, os.path.join(model_dir, "training_columns.joblib"))

print("Saved model and training columns.")
##########################################

model = joblib.load("models/final_model.joblib")
training_columns = joblib.load("models/training_columns.joblib")

print("Loaded local model successfully")

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Bank Churn Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 700px;
            background: white;
            margin: 40px auto;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #222;
        }
        p {
            color: #555;
            text-align: center;
        }
        form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 25px;
        }
        label {
            display: block;
            font-weight: bold;
            margin-bottom: 5px;
        }
        input, select {
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        .full {
            grid-column: span 2;
        }
        button {
            grid-column: span 2;
            padding: 12px;
            border: none;
            border-radius: 8px;
            background: #1f6feb;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover {
            background: #1558b0;
        }
        .result {
            margin-top: 25px;
            padding: 15px;
            border-radius: 8px;
            background: #eef6ff;
            border: 1px solid #cfe2ff;
            font-size: 18px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Bank Churn Prediction</h1>
        <p>Enter customer details to predict whether the customer is likely to churn.</p>

        <form method="POST" action="/predict-form">
            <div>
                <label>Credit Score</label>
                <input type="number" name="CreditScore" required>
            </div>

            <div>
                <label>Geography</label>
                <select name="Geography" required>
                    <option value="France">France</option>
                    <option value="Germany">Germany</option>
                    <option value="Spain">Spain</option>
                </select>
            </div>

            <div>
                <label>Gender</label>
                <select name="Gender" required>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
            </div>

            <div>
                <label>Age</label>
                <input type="number" name="Age" required>
            </div>

            <div>
                <label>Tenure</label>
                <input type="number" name="Tenure" required>
            </div>

            <div>
                <label>Balance</label>
                <input type="number" step="0.01" name="Balance" required>
            </div>

            <div>
                <label>Number of Products</label>
                <input type="number" name="NumOfProducts" required>
            </div>

            <div>
                <label>Has Credit Card</label>
                <select name="HasCrCard" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div>
                <label>Is Active Member</label>
                <select name="IsActiveMember" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div>
                <label>Estimated Salary</label>
                <input type="number" step="0.01" name="EstimatedSalary" required>
            </div>

            <button type="submit">Predict</button>
        </form>

        {% if result %}
        <div class="result">
            {{ result }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE, result=None)

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

        # Apply same encoding as training
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Align columns with training data
        input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

        prediction = model.predict(input_encoded)[0]

        if prediction == 1:
            result = "Prediction: This customer is likely to churn."
        else:
            result = "Prediction: This customer is likely to stay."

        return render_template_string(HTML_PAGE, result=result)

    except Exception as e:
        return render_template_string(HTML_PAGE, result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)