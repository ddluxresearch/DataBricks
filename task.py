import os
import pandas as pd
import numpy as np
import sys
import argparse
import hypertune

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
from google.cloud import storage



def process_gcs_uri(uri: str) -> (str, str, str, str):
    '''
    Receives a Google Cloud Storage (GCS) uri and breaks it down to the scheme, bucket, path and file
    
            Parameters:
                    uri (str): GCS uri

            Returns:
                    scheme (str): uri scheme
                    bucket (str): uri bucket
                    path (str): uri path
                    file (str): uri file
    '''
    url_arr = uri.split("/")
    if "." not in url_arr[-1]:
        file = ""
    else:
        file = url_arr.pop()
    scheme = url_arr[0]
    bucket = url_arr[2]
    path = "/".join(url_arr[3:])
    path = path[:-1] if path.endswith("/") else path
    
    return scheme, bucket, path, file

    
def model_export_gcs(model, model_dir: str) -> str:
    
    scheme, bucket, path, file = process_gcs_uri(model_dir)
    b = storage.Client().bucket(bucket)
    export_path = os.path.join(path, 'model.pkl')
    blob = b.blob(export_path)
    
    blob.upload_from_string(pickle.dumps(model))
    return scheme + "//" + os.path.join(bucket, export_path)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # Input Arguments
    
    parser.add_argument(
        '--max_depth',
        type = int,
        default = 2
    )
    
    parser.add_argument(
        '--min_samples_leaf',
        type = int,
        default = 3
    )
    
    parser.add_argument(
        '--min_samples_split',
        type = int,
        default = 4
    )
    
    parser.add_argument(
        '--model_dir',
        help = 'Directory to output model and artifacts',
        type = str,
        default = os.environ['AIP_MODEL_DIR'] if 'AIP_MODEL_DIR' in os.environ else ""
    )

    args = parser.parse_args()
    max_depth = args.max_depth
    min_samples_leaf = args.min_samples_leaf
    min_samples_split = args.min_samples_split
    model_dir = args.model_dir
    

    ####****** ----> CHANGE THIS YOUR DATASET LOCATION <---- ********#######
    iris_df = pd.read_csv('')
    
    
    feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
    X = iris_df.loc[:, feature_cols]
    y = iris_df.Species
 
    #split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
    #set a random seed so results will be the same for all of us
    np.random.seed(415)
    
    my_model = tree.DecisionTreeClassifier(max_depth=max_depth, 
                                       min_samples_leaf=min_samples_leaf,
                                       min_samples_split=min_samples_split)
    
    my_model.fit(X_train,y_train)
 
    #get predictions
    predictions = my_model.predict(X_test)
    #output metrics - here we chose precision 
    precision = metrics.precision_score(y_true = y_test, y_pred = predictions, average ='weighted')
    model_export_gcs(my_model, model_dir)

    # output metrics  
    print('Precision: '+ str(precision))   
    print('Training job completed')


    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='precision', metric_value=precision)