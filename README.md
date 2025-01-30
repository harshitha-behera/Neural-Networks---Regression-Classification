# Neural-Networks---Regression-Classification

This repository contains Jupyter notebooks that demonstrate the application of neural networks to two different datasets: the Diabetes Dataset and the California Housing Dataset. Both regression and classification tasks are addressed for each dataset.

## Notebooks

### 1. Neural_Network_Models_Diabetes_dataset.ipynb

Objective:

Regression Task: Predict the quantitative progression of diabetes.

Classification Task: Classify patients into "low progression" or "high progression" based on a progression threshold.

Steps:

. Dataset Preparation: The Diabetes Dataset is loaded using load_diabetes() from sklearn.datasets, scaled using StandardScaler, and split into training and test sets. The data is converted into PyTorch tensors.

. Neural Network Definition: A feedforward neural network with 10 input features, two hidden layers (64 and 32 neurons), and an output layer for regression or binary classification.

. Training: For the regression task, MSELoss is used as the loss function and Adam as the optimizer, trained for 100 epochs. For the classification task, CrossEntropyLoss and Adam are used.

. Evaluation: Evaluation metrics such as MSE, R-squared, accuracy, precision, recall, and F1-score are computed. The performance is visualized using plots of true vs. predicted values and confusion matrix.

2. Neural_Network_Models_California_housing_dataset.ipynb

Objective:

Regression Task: Predict the median house values in California.

Classification Task: Classify houses into "low value" or "high value" based on a value threshold.

Steps:

. Dataset Preparation: The California Housing Dataset is loaded using fetch_california_housing() from sklearn.datasets, scaled using StandardScaler, and split into training and test sets. The data is then converted into PyTorch tensors.

. Neural Network Definition: Similar to the Diabetes dataset, a feedforward neural network is defined with 8 input features and two hidden layers (64 and 32 neurons).

. Training: The model is trained for 100 epochs using MSELoss for regression and CrossEntropyLoss for classification.

. Evaluation: The regression task is evaluated with MSE and R-squared. For the classification task, accuracy, precision, recall, and F1-score are computed, and the confusion matrix is visualized.


#### Requirements

Python 3.x

PyTorch

Scikit-learn

Matplotlib

Usage

Clone the repository or download the Jupyter notebooks.


#### Install the required packages using:

pip install -r requirements.txt

Open the notebooks in Jupyter and run the code cells to train the models and evaluate the results.
