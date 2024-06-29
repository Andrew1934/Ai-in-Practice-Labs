
# 1.	Load and explore the dataset.

import pandas as pd
import numpy as np

file_path = "/workspaces/Ai-in-Practice-Labs/Billionaires Statistics Dataset.csv"
df = pd.read_csv(file_path)

print("Initial Dataframe")
print(df.head()) # Shows changes

# 2.	Clean the dataset and prepare it by handling missing values and encoding categorical variables.

df_cleaned = df.drop(columns=['rank', 'personName','organization','city', 'source', 'countryOfCitizenship', 'latitude_country', 'longitude_country']).dropna() # remove unncessary columns

print("Initial Dataframe")
print(df_cleaned.head()) # Shows changes

df_encoded = pd.get_dummies (df_cleaned, columns=['category','country','industries']) # convert category data to numerical data using on-hot encoding

print("Initial Dataframe")
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





