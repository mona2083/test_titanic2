
import os
import sys
import numpy as np
from pathlib import Path
import mlflow
import pandas as pd
from airflow.decorators import dag, task
# from mlflow.models import infer_signature
from datetime import datetime
import joblib

# current_dir = Path(os.getcwd())
# parent_dir = current_dir.parent
# sys.path.append(f"{parent_dir}/src/main/python")

from python.titanic_preprocessing import TitanicPreprocessing
from python.titanic_training import TitanicTraining
from python.titanic_evaluation import TitanicEvaluation


mlflow.set_tracking_uri("http://127.0.0.1:5000")


train_path = "src/data/train.csv"
test_path = "src/data/test.csv"

@dag(
    dag_id="titanic",
    default_args={
        "owner": "airflow",
        "start_date": datetime(2025, 3, 20),
        "retries": 1,
    },
    schedule_interval='@once',
    catchup=False,
    tags=["titanic"]
)
def titanic_dag():

    @task
    def preprocess_task():
        preprocess = TitanicPreprocessing(train_data_path=train_path, test_data_path=test_path)
        train_df, test_df = preprocess.preprocess_data()
        # return train_df, test_df
        joblib.dump((train_df, test_df), '/tmp/data.pkl')

    @task
    def train_model_task():
        train_df, test_df = joblib.load('/tmp/data.pkl')
        model_dict = TitanicTraining().train_model(train_df)
        evaluate_dict = TitanicEvaluation('src/data/gender_submission.csv').evaluate_model(model_dict, test_df)

        register_model = np.nan
        for model_name, eva in evaluate_dict.items():
            if register_model is np.nan:
                register_model = model_name
            else:
                if evaluate_dict[register_model] < eva:
                    register_model = model_name

        mlflow.set_experiment("test")
        for model_name, eva in evaluate_dict.items():
            with mlflow.start_run(run_name=model_name):
                pyfunc_model_path = f"models/{model_name}"
                
                mlflow.log_metric("accuracy", eva)
                print(f"{model_name}: {eva}")
            
                if model_name == register_model:
                    mlflow.sklearn.log_model(
                        model_dict[model_name], 
                        artifact_path=pyfunc_model_path,
                        # signature=signature,
                        registered_model_name=f"best_model_{model_name}",
                        )
                else:
                    mlflow.sklearn.log_model(model_dict[model_name], artifact_path=pyfunc_model_path)

        

    preprocess = preprocess_task()
    train_model = train_model_task()

    (
        preprocess >> train_model
    )

titanic_dag()

