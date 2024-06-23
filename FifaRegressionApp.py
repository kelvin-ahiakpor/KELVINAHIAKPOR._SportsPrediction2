import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import requests

import streamlit as st
import requests
import pickle as pkl

# Function to download file from URL
def download_file_from_onedrive(url, output_path):
    response = requests.get(url)
    with open(output_path, 'wb') as f:
        f.write(response.content)

# Define the OneDrive URL of the file
onedrive_url = 'https://drive.usercontent.google.com/download?id=1AyuTPmyG2krfC2WDzN5GzT55phSWX6oA&authuser=0'

# Path where you want to save the downloaded file
output_path = './try/FifaRandomForestRegressor.pkl'

# Download the file
download_file_from_onedrive(onedrive_url, output_path)

# Load the model
try:
    with open(output_path, 'rb') as model_file:
        model = pkl.load(model_file)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Load the trained model
with open("/Users/kelvin/Library/CloudStorage/OneDrive-AshesiUniversity/ASHESI/2ND YEAR/SEM 2/AI/assignments/assignment 2/models/FifaRandomForestRegressor.pkl", "rb") as model_file:
    model = pkl.load(model_file)

# # Load the imputer
# with open("./imputer.pkl", "rb") as imputer_file:
#     imputer = pkl.load(imputer_file)

# Load the scaler
with open("/Users/kelvin/Library/CloudStorage/OneDrive-AshesiUniversity/ASHESI/2ND YEAR/SEM 2/AI/assignments/assignment 2/scaler.pkl", "rb") as scaler_file:
    scaler = pkl.load(scaler_file)

# Function to preprocess input data and predict using the trained model
def predict_overall(player_data):
    # Convert input data into a DataFrame
    player_features = pd.DataFrame(player_data, index=[0])

    # Handle the case where release_clause_eur is zero by setting it to value_eur
    player_features.loc[player_features['release_clause_eur'] == 0, 'release_clause_eur'] = player_features['value_eur']

    # Apply logarithm transformations
    player_features['log_wage_eur'] = np.log(player_features['wage_eur'] + 1)
    player_features['log_value_eur'] = np.log(player_features['value_eur'] + 1)
    player_features['log_release_clause_eur'] = np.log(player_features['release_clause_eur'] + 1)
    player_features['passing_dribbling_interaction'] = player_features['passing'] * player_features['dribbling']

    selected_features = ['log_value_eur', 'log_release_clause_eur', 'movement_reactions', 'log_wage_eur', 'potential', 'wage_eur', 'passing_dribbling_interaction', 'value_eur', 'passing']
    player_features = player_features[selected_features]

    # Make predictions
    predicted_rating = model.predict(player_features)

    margin_of_error = 1.96 * 0.44

    lower_bound = predicted_rating - margin_of_error
    upper_bound = predicted_rating + margin_of_error

    confidence_score = 1 / margin_of_error 

#     # Impute missing values using the loaded imputer
#     player_features_imputed = imputer.transform(player_features)

    # Scale the input data using the loaded scaler
    X_scaled = scaler.transform(player_features)

    # Make predictions
    predicted_rating = model.predict(X_scaled)

    margin_of_error = 1.96 * 0.44

    lower_bound = predicted_rating - margin_of_error
    upper_bound = predicted_rating + margin_of_error
    
    confidence_score = 1 / margin_of_error 

    return predicted_rating[0], lower_bound[0], upper_bound[0], confidence_score

# Streamlit UI
st.title('FIFA Player Rating Predictor')

# Sidebar with input fields for features
movement_reactions = st.number_input('Enter Movement Reactions', min_value=0, step=1)
potential = st.number_input('Enter Potential', min_value=0, step=1)
wage_eur = st.number_input('Enter Wage (EUR)', min_value=0, step=1)
release_clause_eur = st.number_input('Enter Release Clause (EUR)', min_value=0, step=1)
value_eur = st.number_input('Enter Value (EUR)', min_value=0, step=1)
passing = st.number_input('Enter Passing', min_value=0, step=1)
dribbling = st.number_input('Enter Dribbling', min_value=0, step=1)

# Prepare player data for prediction
player_data = {
    'movement_reactions': movement_reactions,
    'potential': potential,
    'wage_eur': wage_eur,
    'release_clause_eur': release_clause_eur,
    'value_eur': value_eur,
    'passing': passing,
    'dribbling': dribbling
}

# Predict button
if st.button('Predict Player Rating'):
    # Make prediction
    predicted_rating, lower_bound, upper_bound, confidence_score = predict_overall(player_data)

    # Display results
    st.success(f'Predicted Overall Rating: {predicted_rating:.2f}')
    st.info(f'Confidence Interval: [{lower_bound:.2f}, {upper_bound:.2f}]')
    st.info(f'Confidence Score: {confidence_score:.2f}')
