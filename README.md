# KELVINAHIAKPOR._SportsPrediction
# Assignment 2: FIFA Regression Problem - kelvin.ahiakpor

## Overview

This project aimed to predict FIFA player ratings using machine learning regression techniques. The objective was to develop a robust model capable of accurately forecasting player ratings based on various attributes.

## Data Exploration and Understanding

### Data Loading and Inspection

- **Data Sources:** Utilized player attributes from FIFA series datasets (`players` and `players_22`).
- **Initial Exploration:** Explored dataset dimensions, data types, and summary statistics to understand features and distributions.
- **Visualization:** Generated histograms and correlation matrices to identify significant numeric features and relationships.

### Feature Engineering

- **Transformations:** Applied logarithmic transformations to skewed features like `value_eur`, `wage_eur`, and `release_clause_eur` to normalize distributions.
- **Interactions:** Created interaction features such as `passing_dribbling_interaction` to capture combined effects.
- **Ratio Features:** Derived ratio features (`potential_value_ratio`, `physic_movement_reactions_ratio`, etc.) to highlight player attributes relative to their overall rating.

## Data Preparation

### Data Cleaning and Preprocessing

- **Handling Missing Data:** Imputed missing values in `release_clause_eur` using `value_eur` where applicable.
- **Encoding Categorical Variables:** Employed one-hot encoding for categorical variable `work_rate`.

### Feature Selection

- **Correlation Analysis:** Selected features highly correlated with the target variable (`overall`) for model training.

## Model Development

### Model Selection and Training

- **Models Evaluated:** Trained RandomForestRegressor, XGBRegressor, and GradientBoostingRegressor models.
- **Cross-Validation:** Utilized 5-fold cross-validation to assess model performance, ensuring robustness and generalization.

### Model Evaluation

- **Performance Metrics:** Evaluated models using metrics such as RMSE (Root Mean Squared Error) and R2 Score.
- **Fine-tuning:** Optimized model hyperparameters using RandomizedSearchCV to improve prediction accuracy.

## Model Deployment

### Web Application Development

- **Deployment Platform:** Developed a web application using Streamlit to allow users to predict player ratings interactively.
- **Functionality:** Implemented input fields for user-provided player attributes, leveraging trained models for real-time predictions.

## Model Testing

### External Data Validation

- **Testing with New Data:** Validated model performance using `players_22` dataset, assessing prediction accuracy on unseen data.
- **Evaluation:** Calculated RMSE and R2 Score to measure model effectiveness on new data.

## Conclusion

This project successfully demonstrated the application of machine learning techniques to predict FIFA player ratings. By integrating data exploration, feature engineering, model training, and deployment via Streamlit, the project provided a comprehensive solution for predicting and evaluating player ratings interactively.
