# Databricks notebook source
import os
import requests
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    # Get this URL from the models page
    url = 'https://dbc-ee48c8dd-ea5c.cloud.databricks.com/model/subir_iris_classification/8/invocations'
    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
    ds_dict = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  
    return response.json()

# You can generate a token from the User Settings page (click Settings in the left sidebar) and copy the token 
os.environ["DATABRICKS_TOKEN"] = "dapi65ff59a8653c95fb08fc11e7e251e4ac"

#Read the data, change the location based on the user 
iris_df =  pd.read_csv("iris_data.csv")
#format data columns
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = iris_df.loc[:, feature_cols]
y = iris_df.Species
 
#split the data into train and test sets
_, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)

# Number of rows to predict
num_predictions = 5
served_predictions = score_model(X_test[:num_predictions])
served_predictions



# COMMAND ----------


