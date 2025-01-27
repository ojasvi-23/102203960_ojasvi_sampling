# 102203960_ojasvi_sampling

Credit Card Fraud Detection Using Machine Learning
This project demonstrates how to perform credit card fraud detection using machine learning techniques, specifically applying SMOTE for handling imbalanced data and using a RandomForestClassifier for classification.

Table of Contents
Overview
Installation
Data Preprocessing
Model Training
Evaluation
Running the Code
Overview
This project focuses on predicting fraudulent credit card transactions using machine learning. It uses the following steps:

Data Preprocessing: Balancing the dataset with SMOTE to handle class imbalance.
Model Training: Using RandomForestClassifier to train a model on the balanced dataset.
Evaluation: Evaluating the model's performance using accuracy as the metric.
Installation
To run the project, you will need the following libraries installed:

pandas
scikit-learn
imbalanced-learn
matplotlib (optional for data visualization)

To install these dependencies, run:
bash

pip install pandas scikit-learn imbalanced-learn

Data Preprocessing

Loading the Dataset: The dataset is loaded using pandas.read_csv() from a CSV file (Creditcard_data.csv).
Handling Imbalanced Data: SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the dataset by oversampling the minority class.
Train-Test Split: The balanced dataset is split into training and test sets using train_test_split from sklearn.
Model Training
The model is trained using RandomForestClassifier from sklearn. Random forests are ensemble learning models that build multiple decision trees and merge them to get a more accurate and stable prediction.

Evaluation
The model’s performance is evaluated by calculating the accuracy score on the test dataset.

Running the Code
1. Prepare the Dataset
Make sure you have the Creditcard_data.csv file in the same directory as the script or update the path to the file.

2. Run the Script
Once you have all dependencies installed and the dataset prepared, you can run the Python script to execute the model:

bash

python main.py
This will:

Load the data
Apply SMOTE for balancing the dataset
Split the dataset into training and testing sets
Train a RandomForestClassifier
Evaluate the model’s accuracy
Expected Output:
The script will output the model's accuracy on the test dataset. For example:

makefile

Accuracy: 0.98

License
This project is open-source and available under the MIT License. See the LICENSE file for more details.

