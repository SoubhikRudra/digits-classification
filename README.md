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