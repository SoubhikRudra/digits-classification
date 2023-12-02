from sklearn import datasets, preprocessing, model_selection, linear_model
import itertools
import joblib
import os
import numpy as np
from utils import hyperparameter_tuning, prepare_data_splits

def main():
    # Load  dataset
    digits_data = datasets.load_digits()

    # Flatten 
    n_samples = len(digits_data.images)
    flattened_data = digits_data.images.reshape((n_samples, -1))
    X_data = flattened_data
    y_data = digits_data.target

    # Normalize  data - Final Exam - Q1
    X_data_normalized = preprocessing.normalize(X_data, norm='l2')

    # parameter ranges
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    all_models = {}

    for solver in solvers:
        # Logistic Regression with the current solver
        model = linear_model.LogisticRegression(solver=solver)

        # 5-fold cross-validation for mean and standard of performance
        cv_scores = model_selection.cross_val_score(model, X_data_normalized, y_data, cv=5)
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        print(f"Solver: {solver}, Mean Accuracy: {mean_score}, Std Accuracy: {std_score}")

        # Train the model t
        model.fit(X_data_normalized, y_data)

        # Save the model
        model_name = f"M22AIE206_lr_{solver}.joblib"   
        model_path = os.path.join('/home/soubhikr/mlops/digits-classification/API/', model_name)
        joblib.dump(model, model_path)

        # Store the trained model
        all_models[solver] = model


if __name__ == "__main__":
    main()
