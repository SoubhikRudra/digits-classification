"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import  metrics, svm
from utils import preprocess_data, split_data, train_model, read_digits, split_train_dev_test, predict_and_eval
gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
c_ranges = [0.1, 1, 2, 5, 10]

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

# 1 . Get the dataests
X, y = read_digits()

#3. Data Splitting

#X_train, X_test, y_train, y_test = split_data(X,y, test_size=0.3)
X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, test_size=0.3, dev_size=0.3)

#4. Data Preprocessing
#data = preprocess_data(data)
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)
X_dev = preprocess_data(X_dev)

# Hyperparameter Tuning 
#   - Take all combinations of Gamma and C 
best_acc_so_far = -1
best_model = None
for cur_gamma in gamma_ranges:
    for cur_c in c_ranges:
    #- train the model with each parameter 
    #5 Model Training
        cur_model  = train_model(X_train, y_train, {'gamma': cur_gamma, 'C': cur_c}, model_type="svm")
    #- get some performance metrics on Dev Set 
        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)
    #- Select the Hyper parameters that yields the best performance on Dev sets 
        if cur_accuracy > best_acc_so_far:
            print("New Best Accuracy is :", cur_accuracy)
            best_acc_so_far = cur_accuracy
            optimal_gamma = cur_gamma
            optimal_c = cur_c
            best_model = cur_model
print("Optimal Parameters gamma :", optimal_gamma , "C :", optimal_c)

#5 Model Training 
# model  = train_model(X_train, y_train, {'gamma': optimal_gamma, 'C': optimal_c}, model_type="svm")

# 6  Getting Model Prediction on test set
# 7  Qualitative Sanity check on the prediction 
# 8  Model Evaluation
test_accuracy = predict_and_eval(best_model, X_test, y_test)
print("Test Accuracy:",test_accuracy)
# print(
#     f"Classification report for classifier {model}:\n"
#     f"{metrics.classification_report(y_test, predicted)}\n"
# )

# ###############################################################################
# # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# # true digit values and the predicted digit values.

# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")

# # plt.show()

# # The ground truth and predicted lists
# y_true = []
# y_pred = []
# cm = disp.confusion_matrix

# # For each cell in the confusion matrix, add the corresponding ground truths
# # and predictions to the lists
# for gt in range(len(cm)):
#     for pred in range(len(cm)):
#         y_true += [gt] * cm[gt][pred]
#         y_pred += [pred] * cm[gt][pred]

# print(
#     "Classification report rebuilt from confusion matrix:\n"
#     f"{metrics.classification_report(y_true, y_pred)}\n"
# )