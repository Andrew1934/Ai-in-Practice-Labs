
# 1.	Load and explore the dataset.

import pandas as pd
import numpy as np

file_path = "/workspaces/Ai-in-Practice-Labs/Billionaires Statistics Dataset.csv"
df = pd.read_csv(file_path)

print("Initial Dataframe")
print(df.head()) # Shows intial headers

# 2.	Clean the dataset and prepare it by handling missing values and encoding categorical variables.

df_cleaned = df.drop(columns=['rank', 'personName','organization','city', 'source', 'countryOfCitizenship', 'latitude_country', 'longitude_country','status','gender','birthDate','lastName','firstName','title','date','state','residenceStateRegion']).dropna() # remove unncessary columns

df_cleaned['gdp_country'] = df_cleaned['gdp_country'].replace('[\$,]', '', regex=True).astype(float) # Handle formatting of currency in gdp data column.

df_encoded = pd.get_dummies (df_cleaned, columns=['category','country','industries']) # convert category data to numerical data using on-hot encoding

print("Dataframe after preparing")
print(df_encoded.head()) # Shows changes

X = df_encoded.drop('finalWorth', axis=1) # This separates the features and the target variable (finalWorth)
Y = df_encoded['finalWorth']

# 3.	Split the dataset into a training set and a test set.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # Split data into Training and Testing sets.

# 4.	Train a regression model on the training data.

model = LinearRegression() # initialise linear regression model
model.fit(X_train, Y_train) # Train the model on training data

# 5.	Evaluate the model's performance on the test data.

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]} # Define the parameter grid

ridge_model = Ridge() # Initialize GridSearchCV
grid_search = GridSearchCV(estimator=ridge_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

grid_search.fit(X_train, Y_train) # Perform hyperparameter tuning

best_params = grid_search.best_params_ # Get the best parameters
print(f"Best parameters: {best_params}")

best_ridge_model = grid_search.best_estimator_ # Use the best model

# 6.	Perform hyperparameter tuning to improve the model if necessary.

from sklearn.metrics import mean_squared_error
Y_pred = best_ridge_model.predict(X_test) # Predict on the test set

mse = mean_squared_error(Y_test, Y_pred) # Calculate and print the mean squared error
print(f"Mean Squared Error: {mse}")



