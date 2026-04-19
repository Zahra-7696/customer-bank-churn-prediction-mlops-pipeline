from pathlib import Path

def test_model_file_exists():
    assert Path("models/final_model.joblib").exists()

def test_training_columns_file_exists():
    assert Path("models/training_columns.joblib").exists()

def test_app_file_exists():
    assert Path("src/api/app.py").exists()
