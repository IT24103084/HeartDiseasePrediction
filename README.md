## Live Demo

You can try the deployed app here:

[Heart Disease Prediction System] (https://heartdiseasepredictiongit-hfxo8fsd9tungesxsfwjhw.streamlit.app/)

# Heart Disease Prediction System

This is a beginner-friendly machine learning project that predicts the possibility of heart disease based on patient health data.

## Project Overview

The main goal of this project is to analyze heart disease-related health data and build a machine learning model that can predict whether a person has a higher or lower possibility of heart disease.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Dataset

The dataset contains health-related features such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate
- Exercise induced angina
- Oldpeak
- Slope
- Number of major vessels
- Thalassemia
- Target

The target column represents:

- 0: Lower possibility of heart disease
- 1: Higher possibility of heart disease

## Project Workflow

1. Loaded the dataset
2. Checked missing values
3. Performed exploratory data analysis
4. Visualized target distribution and correlations
5. Split data into training and testing sets
6. Applied feature scaling
7. Trained multiple machine learning models
8. Compared model performance
9. Saved the best model
10. Built a Streamlit web app for prediction

## Machine Learning Models Used

- Logistic Regression
- K-Nearest Neighbors
- Random Forest Classifier

## How to Run This Project

Install the required libraries:

```bash
pip install -r requirements.txt