# Fitness Plan Prediction with Random Forest and Pretrained Model

## Overview

This project includes two approaches to training and predicting fitness plans using machine learning models in Python. The goal is to predict a personalized fitness plan based on user inputs such as age range, body type, goal, and fitness level.

1. **Training a Random Forest Model**: In the first approach, we train a Random Forest classifier on fitness-related data and save the model, label encoders, and feature columns.
2. **Loading and Using the Pretrained Model**: In the second approach, we load the saved model, label encoders, and feature columns to predict a fitness plan based on new user inputs.

## Requirements

Before running the program, make sure you have the necessary Python packages installed:

- `pandas`
- `scikit-learn`
- `pickle`

Install these dependencies using pip:

```bash
pip install pandas scikit-learn pickle
```

## Training the Random Forest Classifier
* Place your CSV file at data/data.csv.
* Run the script to train the model and save it:

```bash
python train_model.py
```

## Using the Pretrained Model for Prediction
* Ensure the trained model, label encoder, and columns are saved in the model/ directory.
* Run the script to predict a fitness plan based on new user input:

```bash
python predict_workout.py
```

