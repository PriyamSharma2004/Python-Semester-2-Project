# Python-Semester-2-Project
in this project I have used the wine quality dataset to make a prediction model
Wine Quality Predictor
Overview

This project is a machine learning application that predicts wine quality classes ("Good" or "Bad") based on various physicochemical properties. The model was trained on a dataset containing both red and white wine samples.

Features

Data Processing: Handles missing values, duplicates, and outliers
Exploratory Data Analysis: Includes visualizations of quality distribution, correlation matrices, and statistical tests
Machine Learning Model: Random Forest classifier for quality prediction
Web Interface: Streamlit app for interactive predictions
Deployment: Includes ngrok configuration for temporary public hosting
Dataset

The model uses two datasets combined:

winequality-red.csv - Red wine samples
winequality-white.csv - White wine samples
Original data contains these features:

Fixed acidity
Volatile acidity
Citric acid
Residual sugar
Chlorides
Free sulfur dioxide
Total sulfur dioxide
Density
pH
Sulphates
Alcohol
Quality (target variable)


The web interface allows you to:

Adjust sliders for each wine feature
Click "Predict Quality Class" button
View the predicted class ("Good" or "Bad") and probability breakdown
Model Files

The application requires these files:

model.joblib - Trained Random Forest model
features.joblib - List of feature names in correct order


wine_predictor-2.py - Main Python script containing data processing, EDA, and model training
app.py - Streamlit web application
cleanedwinedata.csv - Processed dataset after cleaning
License

This project is open source, available under the MIT License.

Acknowledgments

Dataset source: UCI Machine Learning Repository
Built using Python's data science ecosystem (pandas, scikit-learn, matplotlib, seaborn)
