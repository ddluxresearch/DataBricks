import json
import pandas as pd
import numpy as np
import sys
 
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import skopt
import argparse
import pickle

# Train a model with these hyperparams: (max_depth, min_samples_leaf, min_samples_split):
def train_with_params(search_params): 

    params = {**search_params}
   
    my_model = tree.DecisionTreeClassifier(max_depth=params['max_depth'], min_samples_leaf= params['min_samples_leaf'], min_samples_split= params['min_samples_split'])
    
    my_model.fit(X_train,y_train)

    #determine predictions
    predictions = my_model.predict(X_test)

    #output metrics - here we chose precision and recall
    precision = metrics.precision_score(y_true = y_test, y_pred = predictions, average ='weighted')
    recall = metrics.recall_score(y_true = y_test, y_pred = predictions, average ='weighted')
    f1_score = metrics.f1_score(y_true = y_test, y_pred = predictions, average ='weighted')
    
    return -precision


if __name__ == '__main__':
    
    #####****** ----> CHANGE THIS YOUR DATASET LOCATION <---- ********#######
    input_file ="/repos/UXResearchRepo/iris_data.csv"

    iris_df = pd.read_csv(input_file, header = 0)
 
    #format data columns
    feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
    X = iris_df.loc[:, feature_cols]
    y = iris_df.Species
 
    #split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
    #set a random seed so results will be the same for all of us
    np.random.seed(415)
    
    parser = argparse.ArgumentParser()
    # Input Arguments
    
    parser.add_argument(
        '--max_depth',
        type = int,
        default = 4
    )
    
    parser.add_argument(
        '--min_samples_leaf',
        type = int,
        default = 4
    )
    
    parser.add_argument(
        '--min_samples_split',
        type = int,
        default = 5
    )
    
    parser.add_argument(
        '--hpo',
        action ='store_true',
        default = False
    )

    args = parser.parse_args()
    max_depth = args.max_depth
    min_samples_leaf = args.min_samples_leaf
    min_samples_split = args.min_samples_split
    hpo = args.hpo


if hpo:
    
    SPACE = [
    skopt.space.Integer(2, 4, name='max_depth'),
    skopt.space.Integer(2, 5, name='min_samples_leaf'),
    skopt.space.Integer(2, 5, name='min_samples_split')]
    
    @skopt.utils.use_named_args(SPACE)
    def objective(**params):
        return train_with_params(params)
    
    HPO_params = {
              'n_calls':20,
              'n_random_starts':20,
              'base_estimator':'ET',
              'acq_func':'EI',
             }

    result = skopt.forest_minimize(objective, SPACE, **HPO_params)
    print(f'Best precision : {-result.fun}\n')
    print(f'Best parameters :\nmax_depth: {result.x[0]:.0f}\nmin_samples_leaf: {result.x[1]:.0f}\nmin_samples_split: {result.x[2]:.0f} \n')
    
    #output metrics to dominostats for display in jobs tab
    with open('dominostats.json', 'w') as f:
        f.write(json.dumps({"Precision": -result.fun, "max_depth": str(result.x[0]), "min_samples_leaf": str(result.x[1]), "min_samples_split":str(result.x[2])}))
        
else:
    
    my_model = tree.DecisionTreeClassifier(max_depth=max_depth, 
                                       min_samples_leaf=min_samples_leaf,
                                       min_samples_split=min_samples_split)
    
    my_model.fit(X_train,y_train)
 
    #get predictions
    predictions = my_model.predict(X_test)
    #output metrics - here we chose precision 
    precision = metrics.precision_score(y_true = y_test, y_pred = predictions, average ='weighted')
    #Save the model to a pickle file
    filename = 'DecisionTree_model.sav'
    pickle.dump(my_model, open(filename, 'wb'))
    print(f'Precison : {precision}')
    
