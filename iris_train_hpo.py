# Databricks notebook source
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import numpy as np
import mlflow
import mlflow.sklearn
import cloudpickle
from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from mlflow.tracking import MlflowClient
import time

from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env


# COMMAND ----------

# Choose the mode in the widget displayed above, this is also used as a parameter in workflows
dbutils.widgets.dropdown("mode", "Train", ["Train", "HPO"])

# COMMAND ----------

mode = dbutils.widgets.get("mode").lower()

# COMMAND ----------

#   ------- UPDATE THE DATA LOCATION -------------- 

#Use this if the file is in the FileStore
# iris_df = spark.read.csv('', header=True, inferSchema=True)

#Use this if the file is there as a table
iris_df = spark.read.format("delta").load('/user/hive/warehouse/iris_data')
iris_df = iris_df.toPandas()

#format data columns
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = iris_df.loc[:, feature_cols]
y = iris_df.Species
 
#split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
#set a random seed so results will be the same for all of us
np.random.seed(415)

#    ------- UPDATE THE MODEL NAME HERE -------------- 
model_name = ""
client = MlflowClient()

# COMMAND ----------

# The predict method of sklearn's DecisionTreeClassifier returns a binary classification (0 or 1). 
# The following code creates a wrapper function, SklearnModelWrapper, that uses 
# the predict_proba method to return the probability that the observation belongs to each class. 

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:,1]

# COMMAND ----------

# Hyper parameters for training
if mode == 'train':
    
    # Let's go with these parameters for now and check the precision
    max_depth = 2
    min_samples_leaf = 2
    min_samples_split = 5
    
    # Change the user name in the path below
    mlflow.set_experiment("/Users/ddluxresearch@gmail.com/iris_training")


        
    with mlflow.start_run(run_name='untuned_decision_tree'):
        my_model = tree.DecisionTreeClassifier(max_depth=max_depth, 
                                           min_samples_leaf=min_samples_leaf,
                                           min_samples_split=min_samples_split)
        my_model.fit(X_train,y_train)

        #get predictions
        predictions = my_model.predict(X_test)
        #output metrics
        precision = metrics.precision_score(y_true = y_test, y_pred = predictions, average ='weighted')
        recall = metrics.recall_score(y_true = y_test, y_pred = predictions, average ='weighted')
        f1_score = metrics.f1_score(y_true = y_test, y_pred = predictions, average ='weighted')
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1_score)
        print(f'Precison : {precision}')
        #Log the model with a signature that defines the schema of the model's inputs and outputs. 
        #When the model is deployed, this signature will be used to validate inputs.
        wrappedModel = SklearnModelWrapper(my_model)
        signature = infer_signature(X_train, wrappedModel.predict(None, X_train))

        # The necessary dependencies are added to a conda.yaml file which is logged along with the model for serving
        conda_env =  _mlflow_conda_env(
                additional_conda_deps=None,
                additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
                additional_conda_channels=None,
            )

        mlflow.pyfunc.log_model("decision_tree_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)
            
        # Get the run id to register the model
        run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_decision_tree"').iloc[0].run_id
        


# COMMAND ----------

if mode == 'hpo':
    # Change the user name in the path below
    mlflow.set_experiment("/Users/ddluxresearch@gmail.com/iris_hpo")
    # Set up the search space for the hyperparameters
    search_space = {
      'max_depth': scope.int(hp.quniform('max_depth', 2, 5, 1)),
      'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 2, 5, 1)),
      'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 5, 1)),
      'seed': 123, # Set a seed for deterministic training
    }

    def train_model(params):
        with mlflow.start_run(nested=True):
            my_model = tree.DecisionTreeClassifier(max_depth=params['max_depth'], min_samples_leaf= params['min_samples_leaf'], min_samples_split= params['min_samples_split'])
            my_model.fit(X_train,y_train)
            predictions = my_model.predict(X_test)
            #output metrics - here we chose to optimize precision 
            precision = metrics.precision_score(y_true = y_test, y_pred = predictions, average ='weighted')
            recall = metrics.recall_score(y_true = y_test, y_pred = predictions, average ='weighted')
            f1_score = metrics.f1_score(y_true = y_test, y_pred = predictions, average ='weighted')
            # Log the metrics
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1_score', f1_score)
            #Log the model
            wrappedModel = SklearnModelWrapper(my_model)
            signature = infer_signature(X_train, wrappedModel.predict(None, X_train))

            # The necessary dependencies are added to a conda.yaml file which is logged along with the model for serving
            conda_env =  _mlflow_conda_env(
                additional_conda_deps=None,
                additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
                additional_conda_channels=None,
            )

            mlflow.pyfunc.log_model("decision_tree_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)
            # Set the loss to -1*precision so fmin maximizes the precision
            return {'status': STATUS_OK, 'loss': -1*precision, 'dtree': my_model.tree_}


    # Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
    # A reasonable value for parallelism is the square root of max_evals.
    spark_trials = SparkTrials(parallelism=10)


    # Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
    # run called "xgboost_models" .
    with mlflow.start_run(run_name='hpo_decision_tree_models'):
        best_params = fmin(
        fn=train_model, 
        space=search_space, 
        algo=tpe.suggest, 
        max_evals=5,
        trials=spark_trials,
      )

    best_run = mlflow.search_runs(order_by=['metrics.precision DESC']).iloc[0]
    print(f'Precision of Best Run: {best_run["metrics.precision"]}')
    run_id = best_run.run_id


# COMMAND ----------

# Register the model 
# If you see the error "PERMISSION_DENIED: User does not have any permission level assigned to the registered model", 
# the cause may be that a model already exists. Try using a different name in model_name
model_version = mlflow.register_model(f"runs:/{run_id}/decision_tree_model", model_name, tags = {"mode": mode})

# COMMAND ----------


