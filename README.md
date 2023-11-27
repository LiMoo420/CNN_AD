# Project For Classifying Alzheimer's Disease

## Author:

mc36455@um.edu.mo For Mr.Bo Yin

mc36588@um.edu.mo For Ms.Zihan Xue

## Abstract

Alzheimer's disease is one of the well-known causes of death in the elderly. Conventional methods are difficult to diagnose the disease at an early stage. Machine learning methods are one of the best options for improving diagnostic accuracy and performance. The heterogeneous dimensions and structure among disease data complicate the diagnostic process. Therefore, appropriate features are needed to address this complexity. In this study, the proposed method is introduced in two main steps.

Firstly, we used traditional machine learning methods to analyze the clinical data: logistic regression, KNN classifier, decision tree/random forest, etc., to conduct correlation analysis and model training and prediction on the indicators of patients in the clinical data.

Then, this project defines a more complex convolutional neural network to preprocess 2D brain MRI image data: The Alzheimer's disease dataset (total 6400) from Kaggle website was used to classify the Alzheimer's disease into four categories according to the degree of pathology: no dementia, very mild dementia, mild dementia and severe dementia. Then, the training, testing and validation sets were divided for model training and prediction.

The results showed that the convolutional neural network had the highest accuracy for the four pathological classifications while obtaining the lowest loss values. At the same time, the sensitivity and specific sensitivity for all cases also confirmed that the method was reliable in the early diagnosis of AD, with less error when detecting the normal state.

## Environmental requirements

python==3.10.9

torch==1.13.1+cu116

matplotlib==3.8.0

torchvision==0.14.1+cu116

pillow==9.3.0

numpy==1.24.1

pandas==2.1.2

seaborn==0.13.0

split-folders==0.5.1

scikit-learn==1.3.2

plotly==5.18.0

scipy==1.11.3

xgboost==2.0.2

catboost==1.2.2

lightgbm==3.3.5

## Construction

### method

* cnn.py: Define the structure, training, validation and prediction functions of convolutional neural networks.
* io.py: Used to read the local data set, preprocess the image (size normalization, image enhancement, array conversion, etc.), and convert the image classification label using the unique thermal coding, and finally return the image array and image classification label of all the data sets in the path.
* plot_feature.py: Plot the first convolutional layer image feature output of the convolutional neural network.
* split.py: Divide the local data set raw_data folder containing four pathologic classifications into training, testing, and validation sets.
* tune.py: Used to conduct a hyperparameter optimization for  cnn model (Not yet finished).

### out

This folder contains all the output from the model, including:

* Accuracy on training and validation sets
* Confusion matrix for test set predictions
* The loss change of the test
* Predictions on the test set
* Convolutional neural network model structure file
* The first convolutional layer feature map output
* Test set labels

### paint.ipynb

* Used to paint line chart of loss change

### tradition_analyze.ipynb

* It contains the analysis, training and prediction of a variety of traditional machine learning methods on clinical data sets of Alzheimer's disease, and output prediction results, confusion matrix, accuracy, etc

### train.py

* This includes hyperparameter definition, dataset loading and partitioning, model definition and training, model prediction and result saving
