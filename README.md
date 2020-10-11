
# CS_7641_Supervised_Learning

The code for this assignment can be found at https://github.com/cliang44/CS_7641_Randomized_Optimization

## Environment

1. This project is run under Jython Environment in Eclipse PyDec
2. This project used ABAGAIL package with some modifications
3. The output folders (NNOUTPUT, CONTPKS, FLIPFLOP, TSP) need to be created in the same folder as the Jython code
4. The NN-plot.ipynb and OR-plot.ipynb run in jupyternotebook and generates output plots.

## Data for NN

Data is preprocessed and splits into trest, train and validation sets randomly. 
Data files are seizure_recog_test.csv for test, seizure_recog_train.csv for train and seizure_recog_val.csv for validation.

## Note for NN problem
In order to get the `NN-*` scripts to run on different sets of hidden layers, you need to change the script to get either 10 nodes per hidden layer or 50 nodes per hidden layer. 

## Execution 

For the neural network problem, run:
 - NN-Backprop.py
 - NN-GA.py
 - NN-RHC.py
 - NN-SA.py
 
 For the three randomized optimization problems, run:
 - continuoutpeaks.py
 - tsp.py
 - flipflop.py




