# Predicting-DNA-repair-Outcomes
Machine learning pipeline for predicting DNA repair outcomes.
# DNA Repair Outcome Prediction

This project implements a pipeline to predict DNA repair outcomes using **One-Hot Encoded Guide Sequences** and machine learning models (Stacking).
## Overview
The workflow involves:
1. **One-Hot Encoding of DNA Sequences**:
   - DNA sequences are converted into numerical representations using a one-hot encoding scheme.
   - This enables the use of DNA data in machine learning models.
2. **Multi-Output Regression**:
   - Trains separate models for each target variable.
3. **Data Splitting**:
   - Data is split into training and validation sets to evaluate model performance.
4. **Model Training and Evaluation**:
   - Gradient Boosting Regressors are used to predict multiple target variables.
   - Model performance is measured using the R² (coefficient of determination) score.
