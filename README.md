System Requirement :
    OS
    H/W -- May be skipped

How to Set up  : 
    install conda

    conda create -n digit python=3.9
    conda deactivate
    conda activate digit
    pip install -r requirements.txt

How to Run :  
    python exp.py

Places of Randomness :
    1. Creating the Split 
        - Freezing the data (Shuffle while splitting the test and train data)
    1.5 -- Data order (Learning is iterative)
    2. Model 
        - Weight Initialization 

Meaning of Failure  :
    Poor Performance Metrics
    Coding Runtime/compile Error 
    The model provided bad prediction on new test samples

Feature : 
    Vary Model Hyper Parameter


e.g. 100 samples  : 2-class/binary  classification: image of carrot or turnip
    50 samples  : Carrots
    50 samples  : Turinp
        Data Distribution  : Balanced/Uniform
    
    x amount of data for training
    n-x amount of data for testing  

    Calculate some eval metric(train model(70 samples in training  - 36 carrot and 35 turnip ) , 30 samples in testing 15,15) ==performance

In Practice : 
    Train - Training the model   (Model Type, Hyper parameters, interations)
    Development/Validation  - Selecting the model 
    Testreporting the performance