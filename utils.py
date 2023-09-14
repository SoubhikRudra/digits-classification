from sklearn import  svm, datasets
from sklearn.model_selection import train_test_split
# Put the Utils here 
def read_digits():
    digits = datasets.load_digits()
    X= digits.images
    y= digits.target
    return X, y

def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into 50% train and 50% test subsets
def split_data(x,y,test_size,random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# Train the model of choice with model parameter 
def train_model(x,y, model_params, model_type="svm"):
    if model_type == "svm":
        # Create a classifier: a support vector classifier
        clf = svm.SVC;    
    model = clf(**model_params)
    #Train the model
    model.fit(x,y)
    return model

#Assignment2 - Added below functions
#Added spaces to capture pull request for Assignemnt2 documentation
def split_train_dev_test(X, y, test_size, dev_size, random_state=1):
    # First, split data into training and temporary test subsets
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Next, split the remaining data (X_train_dev, y_train_dev) into training and development subsets
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_dev, y_train_dev, test_size=dev_size, random_state=random_state
    )
    
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def predict_and_eval(model, X_test, y_test):
    # Predict the value of the digit on the test subset
    predicted = model.predict(X_test)

    # Quantitative sanity check
    return predicted