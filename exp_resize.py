"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

import cv2
import numpy as np

from utils import preprocess_data, read_digits, predict_and_eval, train_test_dev_split, tune_hyper_parameters
gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]

C_ranges = [0.1, 1, 2, 5, 10]

param_combinations = [{"gamma": gamma, "c_range": C} for gamma in gamma_ranges for C in C_ranges]

# 1. Get the dataset
X, y = read_digits()
print(X.shape)

print("\n","="*40,"quiz question","="*40,"\n")
print(f"[info] Total no of samples in dataset(TRAIN + TEST + DEV): [{len(y)}]")
print(f"[info] Size of Individual Images: [{X[0].shape[0]} x {X[0].shape[1]}] Pixels")
print("\n","="*40,"Quiz Ended","="*40,"\n")
c = 0
# for test_size in [0.1, 0.2, 0.3]:
    # for dev_size in [0.1, 0.2, 0.3]:

test_size, dev_size = 0.2, 0.1

resize_value = [(4,4), (6,6), (8,8)]



for resize_factor in resize_value:
    c += 1
    print("\n\n","="*30,f"Start the Experiment: {c}","="*30)
    
    X, y = read_digits()
    resized_X = np.empty((1797, resize_factor[0], resize_factor[1]))
    for idx, img in enumerate(X):
        resized_X[idx] = cv2.resize(img, resize_factor, interpolation=cv2.INTER_AREA)
    X = resized_X
    print(f"[INFO] Resized the images to [{resize_factor[0]} x {resize_factor[1]}] Pixels")
    X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size, dev_size)
    X_train, X_test, X_dev = preprocess_data(X_train), preprocess_data(X_test), preprocess_data(X_dev)
    
    best_model, best_accuracy, best_gamma, best_c = tune_hyper_parameters(X_train, y_train, X_dev, y_dev, param_combinations)

    print(f"test_size: [{test_size}]\ndev_size: [{dev_size}]\ntrain_size: [{1-test_size-dev_size}]\noptimal_gamma: [{best_gamma}]\noptimal_c: [{best_c}]\nbest_dev_acc: [{best_accuracy}]")

    # Evaluate 
    test_acc = predict_and_eval(best_model, X_test, y_test)
    print(f"test_acc===> [{test_acc}]")

    test_acc = predict_and_eval(best_model, X_test, y_test)
    print("\n\nTest accuracy: ", test_acc)