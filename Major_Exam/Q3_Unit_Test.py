import joblib
from sklearn.linear_model import LogisticRegression
import os
import pytest

@pytest.mark.parametrize("solver", ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
def test_loaded_model(solver):
    rollno = "M22AIE206"
    model_name = f"{rollno}_lr_{solver}.joblib"
    model_path = os.path.join('/home/soubhikr/mlops/digits-classification/API/', model_name)

    # Load Model
    loaded_model = joblib.load(model_path)

    # if loaded model is the instance of Logistic Regression
    assert isinstance(loaded_model, LogisticRegression) == True
    print(f" ASSERT success - test loaded model validation: {loaded_model}")

@pytest.mark.parametrize("solver", ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
def test_solver_name_match(solver):
    rollno = "M22AIE206"
    model_name = f"{rollno}_lr_{solver}.joblib"
    model_path = os.path.join('/home/soubhikr/mlops/digits-classification/API/', model_name)

    # Load Model
    loaded_model = joblib.load(model_path)

    # If solver name in Model file name matches  solver in Model
    model_solver = loaded_model.get_params()['solver']
    assert solver == model_solver
    print(f"ASSERT - success for test solver name match validation: {loaded_model}")
