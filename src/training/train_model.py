import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "bank_churn_train.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

drop_cols = ["Exited", "RowNumber", "CustomerId", "Surname"]
existing_drop_cols = [col for col in drop_cols if col in df.columns]

X = df.drop(columns=existing_drop_cols)
y = df["Exited"]

X_encoded = pd.get_dummies(X, drop_first=True)
training_columns = X_encoded.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

joblib.dump(rf_model, os.path.join(MODELS_DIR, "final_model.joblib"))
joblib.dump(training_columns, os.path.join(MODELS_DIR, "training_columns.joblib"))

print("Saved model and training columns successfully.")
