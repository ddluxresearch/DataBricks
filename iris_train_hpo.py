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

# Choose the mode in the widget displayed above
dbutils.widgets.dropdown("mode", "Training", ["Training", "HPO"])

# COMMAND ----------

mode = dbutils.widgets.get("mode").lower()

# COMMAND ----------

#Read the data, change the location based on the user 
iris_df =  pd.read_csv("/Workspace/Repos/subir.mansukhani@dominodatalab.com/UXResearchRepo/iris_data.csv")

#format data columns
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = iris_df.loc[:, feature_cols]
y = iris_df.Species
 
#split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
#set a random seed so results will be the same for all of us
np.random.seed(415)

model_name = "iris_classification"
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
if mode == 'training':
    
    max_depth = 2
    min_samples_leaf = 2
    min_samples_split = 5
    # Change the user name in the path below
    mlflow.set_experiment("/Users/subir.mansukhani@dominodatalab.com/iris_training")


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

        mlflow.pyfunc.log_model("untuned_decision_tree_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)

            
        # Register the model in the registry
        run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_decision_tree"').iloc[0].run_id
        # If you see the error "PERMISSION_DENIED: User does not have any permission level assigned to the registered model", 
        # the cause may be that a model already exists with the name "wine_quality". Try using a different name.

        model_version = mlflow.register_model(f"runs:/{run_id}/untuned_decision_tree_model", model_name)

        # Registering the model takes a few seconds, so add a small delay
        time.sleep(15)

        # Tag the model to production
        client.transition_model_version_stage(
          name=model_name,
          version=model_version.version,
          stage="Production",
        )
    
# Print the feature importance for the model
feature_importances = pd.DataFrame(my_model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)


# COMMAND ----------

if mode == 'hpo':
    # Change the user name in the path below
    mlflow.set_experiment("/Users/subir.mansukhani@dominodatalab.com/iris_hpo")
    search_space = {
      'max_depth': scope.int(hp.quniform('max_depth', 2, 5, 1)),
      'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 2, 5, 1)),
      'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 5, 1)),
      'seed': 123, # Set a seed for deterministic training
    }

    def train_model(params):
        # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
        mlflow.sklearn.autolog()
        with mlflow.start_run(nested=True):
            my_model = tree.DecisionTreeClassifier(max_depth=params['max_depth'], min_samples_leaf= params['min_samples_leaf'], min_samples_split= params['min_samples_split'])
            my_model.fit(X_train,y_train)
            predictions = my_model.predict(X_test)
            #output metrics - here we chose precision 
            precision = metrics.precision_score(y_true = y_test, y_pred = predictions, average ='weighted')
            recall = metrics.recall_score(y_true = y_test, y_pred = predictions, average ='weighted')
            f1_score = metrics.f1_score(y_true = y_test, y_pred = predictions, average ='weighted')
            # Log the metrics
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1_score', f1_score)
            #Log the model
            signature = infer_signature(X_train, my_model.predict(X_train))
            mlflow.sklearn.log_model(my_model, "model", signature=signature)
            # Set the loss to -1*precision so fmin maximizes the precision
            return {'status': STATUS_OK, 'loss': -1*precision, 'dtree': my_model.tree_}


    # Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
    # A reasonable value for parallelism is the square root of max_evals.
    spark_trials = SparkTrials(parallelism=10)


    # Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
    # run called "xgboost_models" .
    with mlflow.start_run(run_name='decision_tree_models'):
        best_params = fmin(
        fn=train_model, 
        space=search_space, 
        algo=tpe.suggest, 
        max_evals=10,
        trials=spark_trials,
      )

    best_run = mlflow.search_runs(order_by=['metrics.precision DESC']).iloc[0]
    print(f'Precision of Best Run: {best_run["metrics.precision"]}')
    
    # Update the production iris_classification model in MLflow Model Registry
    new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", model_name)
    # Registering the model takes a few seconds, so add a small delay
    time.sleep(15)

    # Tag the new model version as Production
    client.transition_model_version_stage(
      name=model_name,
      version=new_model_version.version,
      stage="Production"
    )

# COMMAND ----------


