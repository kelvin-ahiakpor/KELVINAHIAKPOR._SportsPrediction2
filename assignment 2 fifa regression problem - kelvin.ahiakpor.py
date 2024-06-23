#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: FIFA Regression Problem - kelvin.ahiakpor

# ### Imports

# In[1]:


import os
import pandas as pd
import numpy as np
import pickle as pkl
import streamlit as st
import matplotlib.pyplot as plt

from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor
from joblib import Parallel, delayed


# ### Setting job timeout for computation

# In[2]:


os.environ['JOBLIB_START_METHOD'] = 'loky'
os.environ['JOBLIB_TIMEOUT'] = '300'


# ### Loading data

# In[3]:


players = pd.read_csv('male_players (legacy).csv', low_memory=False)
players_22 = pd.read_csv('players_22.csv', low_memory=False)


# ### Understanding the data

# ##### Custom recipes for Data Loading and Inspection

# In[4]:


pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 3) 


# ##### Peek at training data

# In[5]:


print(f"Dimensions of data set is {players.shape} This is a large dataset.\n")
players.info()
players.head()


# ##### Quick statistics
# Extended Five-number Summary,
# Histogram Plots,
# Correlation Matrix,
# Categorical Variable Selection &
# Strongly Correlated Numeric Features

# In[6]:


print("Extended Five-number Summary")
players.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
players.hist(bins=50, figsize=(23,20))
print("Histogram Plots")
plt.show()


# In[8]:


print("Correlation Matrix")
quant_players = players.select_dtypes(include=[np.int64, np.float64])
corrMat = quant_players.corr()
corrMat


# In[9]:


print("Categorical Variable Selection")
cat_players = players.select_dtypes(include=[object])
cat_players = cat_players[["work_rate"]]

for column in cat_players.columns:
    print(cat_players[column].value_counts())
    print()
    
cat_players.head()


# In[10]:


print("Most Correlated Numeric Features")
most_correlated = corrMat["overall"].abs()[corrMat["overall"].abs() > 0.50].sort_values(ascending=False)
most_correlated_cols = most_correlated.index.tolist()
most_correlated


# ##### Descriptions of the most correlated numeric features from Kaggle
# 1. overall - player current overall attribute
# 2. movement_reactions - player reactions attribute
# 3. potential - player potential overall attribute
# 4. mentality_composure - player composure attribute
# 5. passing - player passing attribute
# 6. wage_eur - player weekly wage (in eur)
# 7. dribbling - player dribbling attribute
# 8. release_clause_eur - player release clause (in eur) - if applicable
# 9. value_eur - player value (in eur)
# 10. physic - player physic attribute

# #### Some notes so far
# 
# 1. Possible new features for feature engineering:   
# 
#     **Transformation Features** 
#     - log_value_eur - the histogram plots show player value (`value_eur`) is heavily right skewed. Taking the logarithm helps us normalize.
#     - log_wage_eur - the histogram plots show wages (`wage_eur`) is heavily right skewed. Taking the logarithm helps us normalize the distribution.
#     - log_release_clause_eur - the histogram plots show release clause (`release_clause_eur`) is heavily right skewed. Try normalizing with log.
# 
#     **Interaction Features** 
#     - passing_dribbling_interaction - helps us understand the combined effect of passing and dribbling on overall rating.
# 
#     **Ratio Features**
#     - potential/value_eur ratio - a higher ratio may indicate that the player is undervalued relative to their potential (`potential`).
#     - physic/movement_reactions ratio - a balanced ratio suggests a well-rounded player, while extremes may highlight specialized roles or weaknesses.
#     - age vs potential ratio - with this ratio younger players are expected to have higher growth potential (`potential`). *
#     - wage vs value ratio - this ratio shows how the value of a player's contributions (`wage_eur`) compare to their market value (`value_eur`). *
#     - composure vs reactions ratio - better players make good decisions under pressure. A balanced ratio indicates a calm but swift player.
#     - passing vs dribbling ratio - a higher ratio suggests a player is more proficient in distributing the ball relative to individual ball-handling skills.
# 
#     These ratios may (arguably*) influence overall player rating
#     * age vs potential ratio - some players have an older peak age
#     * wage vs value ratio - older players on higher wages may still have lower value 
#     
#     For transformation, we could use exponentiating if a left-skewed distributed was observed
#     
# 
# 2. Imputation:
#    - Since not every player has a release clause, when imputing for N/As in release clause, simply use `value_eur`.
#    - `value_eur` is the best estimation of the player's current or most recent valuation.
#    - Use dataframe joining methods after removing duplicates. Concatenating will defeat the purpose of duplicate removal and add the duplicate rows back.
# 
# 3. Evaluation:
#    - Since the dataset is large (150,000+ rows) use at least 5-fold cross-validation.
#    - Spot checking?
# 

# # Question 1
# Demonstrate the data preparation & feature extraction process

# ### Data Cleaning

# ##### From the quick statistics above, we identified the most relevant numeric columns. Now we pick only those

# In[11]:


df = players[most_correlated_cols]
df


# In[12]:


df[df.duplicated()]


# ##### Duplicates arise because some rows have the same values in these the 10 columns even though the original dataset has no duplicates.

# In[13]:


df = df.drop_duplicates()


# ##### Imputing missing values

# In[14]:


df.loc[df['release_clause_eur'].isnull(), 'release_clause_eur'] = df['value_eur']
#uses boolean series as rows, release clause as column. 
#each null release clause will now have the value of it's respective player's value

imputer_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

with open("./imputer.pkl", 'wb') as f:
    pkl.dump(imputer_pipeline, f)
    
df = pd.DataFrame(imputer_pipeline.fit_transform(df), columns=df.columns)
df


# ### Data Pre-processing

# ##### Encoding categorical features

# In[15]:


encoding_pipeline = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(sparse_output=False),cat_players.columns)
])

cat_players_encoded = encoding_pipeline.fit_transform(cat_players)

df2 = pd.DataFrame(cat_players_encoded, columns=encoding_pipeline.get_feature_names_out())
df2


# ##### Merging the numeric and categorical data
# We remain cautious of the dataframe dimensions. 
# 
# Since we removed duplicates in numeric dataframe we must maintain that row count in the merged dataframe for training. 
# 
# We join on unique column player_id which we will retrieve from players dataframe.

# In[16]:


df.head()


# In[17]:


df2.head() 


# We have to add the player_id to our dataframes first.
# Below we retrieve unique player_id from players into numeric and cateogrical dataframes

# In[18]:


df.loc[:, 'player_id'] = players.loc[df.index, 'player_id'].values
df2.loc[:, 'player_id'] = players.loc[df2.index, 'player_id'].values
#The code below checks if the df has the correct ids from players even though we removed the duplicates initially
#didn't want a case where the id between the cleaned df mismatch the ids in the players
#The duplicates indices were retrieved from running this code: df[df.duplicated()]. 
#Some duplicate indicies to test [2904,161365,3007]

# print(players.iloc[161365]['player_id'])
# print(df.iloc[161365]['player_id'])

# print(players.iloc[161365]['player_id'])
# print(df.iloc[161365]['player_id'])
#Successful!


# Finally, we merge!
# ... and drop player_id.

# In[19]:


playersNeeded = pd.merge(df, df2, on='player_id')
playersNeeded.drop(columns=['player_id'], inplace=True)
playersNeeded


# # Question 2
# Create feature subsets that show maximum correlation with the dependent variable

# ### Feature Engineering

# In[20]:


dfP = playersNeeded
#Transformed Features
dfP['log_wage_eur'] = np.log(dfP['wage_eur'])
dfP['log_value_eur'] = np.log(dfP['value_eur'])
dfP['log_release_clause_eur'] = np.log(dfP['release_clause_eur'])
#Interaction Features
dfP['passing_dribbling_interaction'] = dfP['passing'] * dfP['dribbling']
#Ratio Features
dfP['potential_value_ratio'] = dfP['potential'] / dfP['value_eur']
dfP['physic_movement_reactions_ratio'] = dfP['physic'] / dfP['movement_reactions']
dfP['passing_dribbling_ratio'] = dfP['passing'] / dfP['dribbling']
dfP['wage_value_ratio'] = dfP['wage_eur'] / dfP['value_eur']
dfP['composure_reactions_ratio'] = dfP['mentality_composure'] / dfP['movement_reactions']
dfP.head()


# In[21]:


print("Most Correlated Features")
corrMat = dfP.corr()
most_correlated = corrMat["overall"].abs()[corrMat["overall"].abs() > 0.60].sort_values(ascending=False)
most_correlated_cols = most_correlated.index.tolist()
most_correlated


# The transformed features in our feature engineering came out with higher correlations as compared to the ratio features. 
# 
# The passing_dribbling_interaction also proved useful. 
# 
# Now we will move on, selecting only these 9 features and 1 label to train our model.

# In[22]:


dfP = dfP[most_correlated_cols]
dfP


# ### Feature Scaling

# In[23]:


X = dfP.drop('overall',axis=1)
y = dfP['overall']

pipeline = Pipeline([
    ('scaler', StandardScaler())
])

X_scaled = pipeline.fit_transform(X)
X_scaled

with open("./scaler.pkl", 'wb') as f:
    pkl.dump(pipeline, f)
    
X = pd.DataFrame(X_scaled, columns=X.columns)
X


# # Question 3
# Create and train a suitable machine learning model with cross-validation that can predict a player's rating

# We will train 3 models with cross validation and select the best one. 
# 
# Metrics are models are defined in dictionaries with their inbuilt scorer names
# 
# In our cross validation, we will shuffle the dataset since it is currently ordered from best to worst player

# In[24]:


models = {
    'RandomForest': RandomForestRegressor(random_state=8),
    'XGBoost': XGBRegressor(random_state=8),
    'GradientBoost': GradientBoostingRegressor(random_state=8)
}


# ### Model Training

# In[25]:


kfold = KFold(n_splits=5, random_state=8, shuffle=True)
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2',n_jobs=-1)
    print(f"\n{name} Cross-Validation Scores:")
    print(scores)
    print(f"R2 Avg (sd): {scores.mean() * 100.0:.3f}% ({scores.std() * 100.0:.3f}%)")


# # Question 4
# Measure the model's performance and fine-tune it as a process of optimization

# We will use RMSE to evaluate because it is an interpretable metric in same units as target variable (player rating)

# In[26]:


metrics = {
    'RMSE': 'neg_root_mean_squared_error',
}


# ### Model Evaluation

# In[27]:


for model_name, model in models.items():
    print(f"Evaluation for {model_name}:")
    for metric_name, metric_value in metrics.items():
        scorer = metric_value 
        metric_scores = cross_val_score(model, X_scaled, y, cv=5, scoring=scorer, n_jobs=-1)
        print(f"  {metric_name}: {-metric_scores.mean():.2f} (± {metric_scores.std():.2f})") 


# ### Fine-tuning

# Creating parameter dictionaries

# In[28]:


param_grids = {
    'RandomForest': {
        'n_estimators': randint(50, 200),
        'max_features': [1.0, 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5)
    },
    'XGBoost': {
        'n_estimators': randint(50, 200),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4)
    },
    'GradientBoost': {
        'n_estimators': randint(50, 200),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.6, 0.4),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5)
    }
}


# Tuning all models with Randomized Search and selecting the best parameters

# In[29]:


best_models = {}
for name, model in models.items():
    param_grid = param_grids[name]
    rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=7, 
                                 scoring='neg_root_mean_squared_error', cv=kfold, random_state=8, n_jobs=-1)
    rsearch.fit(X_scaled, y)
    best_models[name] = rsearch.best_estimator_
    print(f"\n{name} Best Parameters: {rsearch.best_params_}")
    print(f"Best RMSE: {-rsearch.best_score_:.2f}")


# ### Re-testing and Re-evaluation

# Cross validation allows us to train and test on the go
# 
# After cross validation we will save our model to a relative directory

# In[30]:


for name, model in best_models.items():
    scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='neg_root_mean_squared_error',n_jobs=-1)
    pkl.dump(model, open("./models/Fifa"+model.__class__.__name__ + ".pkl","wb"))
    print(f"{name} RMSE: {-scores.mean():.2f} (± {scores.std():.2f})")


# # Question 5
# Use the data from another season (players_22) which was not used during the training to test how good is the model

# Select features

# In[31]:


features = ['overall', 'movement_reactions', 'potential', 'wage_eur', 'release_clause_eur', 'value_eur', 'passing', 'dribbling']


# ### Model Testing

# Create function that automates the pipeline
# - Impute
# - Feature Engineer
# - Scale
# - Fit
# - Predict
# - Evaluate

# In[32]:


def preprocess_and_model(players_data):
    # select features
    players_22 = players_data[features].copy()  # Ensure original data is not modified

    # identify columns with NaN values
    nan_columns = players_22.columns[players_22.isna().any()].tolist()

    # impute NaN values with mean
    imputation_dict = {col: players_22[col].mean() for col in nan_columns}
    players_22.fillna(imputation_dict, inplace=True)

    # feature engineering
    players_22['log_wage_eur'] = np.log(players_22['wage_eur'])
    players_22['log_value_eur'] = np.log(players_22['value_eur'])
    players_22['log_release_clause_eur'] = np.log(players_22['release_clause_eur'])
    players_22['passing_dribbling_interaction'] = players_22['passing'] * players_22['dribbling']

    # separate features and label
    X = players_22.drop('overall', axis=1)
    y = players_22['overall']

    # scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # impute any remaining NaNs after feature engineering
    imputer = SimpleImputer(strategy='mean')
    X_final = imputer.fit_transform(X_scaled)

    # fit the RandomForestRegressor model
    model = RandomForestRegressor(random_state=8)
    model.fit(X_final, y)

    # predict with the trained model
    predictions = model.predict(X_final)

    # evaluate
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2_value = r2_score(y, predictions)

    print("Evaluation scores of RandomForestRegressor on players dataset")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2_value*100:.2f}%")


# Call the function

# In[33]:


preprocess_and_model(players_22)


# # Question 6
# Deploy the model on a simple web page using either (Heroku, Streamlite, or Flask) and upload a video that shows how the model performs on the web page/site.

# ### Webpage Creation

# The code below creates the webpage. It has been moved to a .py file to be run on a separate streamlit host

# In[34]:


# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle as pkl
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer

# # Load the trained model
# with open("/Users/kelvin/Library/CloudStorage/OneDrive-AshesiUniversity/ASHESI/2ND YEAR/SEM 2/AI/assignments/assignment 2/models/FifaRandomForestRegressor.pkl", "rb") as model_file:
#     model = pkl.load(model_file)

# # # Load the imputer
# # with open("./imputer.pkl", "rb") as imputer_file:
# #     imputer = pkl.load(imputer_file)

# # Load the scaler
# with open("/Users/kelvin/Library/CloudStorage/OneDrive-AshesiUniversity/ASHESI/2ND YEAR/SEM 2/AI/assignments/assignment 2/scaler.pkl", "rb") as scaler_file:
#     scaler = pkl.load(scaler_file)

# # Function to preprocess input data and predict using the trained model
# def predict_overall(player_data):
#     # Convert input data into a DataFrame
#     player_features = pd.DataFrame(player_data, index=[0])

#     # Handle the case where release_clause_eur is zero by setting it to value_eur
#     player_features.loc[player_features['release_clause_eur'] == 0, 'release_clause_eur'] = player_features['value_eur']

#     # Apply logarithm transformations
#     player_features['log_wage_eur'] = np.log(player_features['wage_eur'] + 1)
#     player_features['log_value_eur'] = np.log(player_features['value_eur'] + 1)
#     player_features['log_release_clause_eur'] = np.log(player_features['release_clause_eur'] + 1)
#     player_features['passing_dribbling_interaction'] = player_features['passing'] * player_features['dribbling']

#     selected_features = ['log_value_eur', 'log_release_clause_eur', 'movement_reactions', 'log_wage_eur', 'potential', 'wage_eur', 'passing_dribbling_interaction', 'value_eur', 'passing']
#     player_features = player_features[selected_features]

# #     # Impute missing values using the loaded imputer
# #     player_features_imputed = imputer.transform(player_features)

#     # Scale the input data using the loaded scaler
#     X_scaled = scaler.transform(player_features)

#     # Make predictions
#     predicted_rating = model.predict(X_scaled)

#     # For demonstration purpose, assuming a fixed margin of error (you can adjust this)
#     margin_of_error = 1.96 * 0.46  # Adjust based on your desired confidence level and RMSE

#     lower_bound = predicted_rating - margin_of_error
#     upper_bound = predicted_rating + margin_of_error

#     return predicted_rating[0], lower_bound[0], upper_bound[0]

# # Streamlit UI
# st.title('FIFA Player Rating Predictor')

# # Sidebar with input fields for features
# movement_reactions = st.number_input('Enter Movement Reactions', min_value=0, step=1)
# potential = st.number_input('Enter Potential', min_value=0, step=1)
# wage_eur = st.number_input('Enter Wage (EUR)', min_value=0, step=1)
# release_clause_eur = st.number_input('Enter Release Clause (EUR)', min_value=0, step=1)
# value_eur = st.number_input('Enter Value (EUR)', min_value=0, step=1)
# passing = st.number_input('Enter Passing', min_value=0, step=1)
# dribbling = st.number_input('Enter Dribbling', min_value=0, step=1)

# # Prepare player data for prediction
# player_data = {
#     'movement_reactions': movement_reactions,
#     'potential': potential,
#     'wage_eur': wage_eur,
#     'release_clause_eur': release_clause_eur,
#     'value_eur': value_eur,
#     'passing': passing,
#     'dribbling': dribbling
# }

# # Predict button
# if st.button('Predict Player Rating'):
#     # Make prediction
#     predicted_rating, lower_bound, upper_bound = predict_overall(player_data)

#     # Display results
#     st.success(f'Predicted Overall Rating: {predicted_rating:.2f}')
#     st.info(f'Confidence Interval: [{lower_bound:.2f}, {upper_bound:.2f}]')


# ### Deployment

# ### Video Plan

# The plan for the video is as follows
# - Find new data that our model has never seen
# - Select data from the top, middle and end (high-low rating)
# - Select players without release clause
# - Concatenate into dataframe for predictions
# - Show predictions for a high, medium, low rated and no release clause player

# #### Loading completely new data

# In[35]:


players_19 = pd.read_csv('players_19.csv', low_memory=False)


# In[36]:


features = ['short_name','overall', 'movement_reactions', 'potential', 'wage_eur', 'release_clause_eur', 'value_eur', 'passing', 'dribbling']
no_release_clause = players_19[pd.isna(players_19['release_clause_eur'])].iloc[0:5]
players_19 = players_19[features]
no_release_clause = no_release_clause[features]


# In[37]:


players_19['release_clause_eur'] = players_19['release_clause_eur'].apply(lambda x: '{:.0f}'.format(x))
players_19['value_eur'] = players_19['value_eur'].apply(lambda x: '{:.0f}'.format(x))
players_19['wage_eur'] = players_19['wage_eur'].apply(lambda x: '{:.0f}'.format(x))


# In[38]:


players_19 = pd.concat([
    players_19.iloc[0:5],
    players_19.iloc[9000:9005],
    players_19.iloc[18080:18085],
    no_release_clause
])


# Dataframe for predictions

# In[39]:


players_19


# # References
# 
# [1]Jim Frost. 2018. Interpreting Correlation Coefficients. Statistics By Jim. Retrieved June 16, 2024 from https://statisticsbyjim.com/basics/correlations/
# 
# [2]Alboukadel Kassambara. 2018. Transform Data to Normal Distribution in R: Easy Guide. Datanovia. Retrieved June 17, 2024 from https://www.datanovia.com/en/lessons/transform-data-to-normal-distribution-in-r/#google_vignette
# 
# [3]Stefano Leone. 2021. FIFA 22 complete player dataset. Kaggle.com. Retrieved June 22, 2024 from https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset?resource=download&select=players_19.csv
# 
# [4]Stefano Leone. 2024. FIFA 23 complete player dataset. www.kaggle.com. Retrieved June 22, 2024 from https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset
# 
# [5]Hrvoje Smolic. 2024. How Much Data Do You Need for Machine Learning. Graphic Note. Retrieved June 16, 2024 from https://graphite-note.com/how-much-data-is-needed-for-machine-learning/#:~:text=The%20rule%2Dof%2Dthumb%20rule
# 
# [6]StackExchange. 2014. regression - What is the reason the log transformation is used with right-skewed distributions? Cross Validated. Retrieved June 17, 2024 from https://stats.stackexchange.com/questions/107610/what-is-the-reason-the-log-transformation-is-used-with-right-skewed-distribution
# 
# [7]StackExchange. 2023. How many features is too many when using feature selection methods? Data Science Stack Exchange. Retrieved June 16, 2024 from https://datascience.stackexchange.com/questions/122640/how-many-features-is-too-many-when-using-feature-selection-methods
