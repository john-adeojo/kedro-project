import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_score, recall_score, f1_score, auc
import matplotlib.pyplot as plt
from ludwig.api import LudwigModel
import requests
import yaml
import json


def read_data(parameters: Dict) -> pd.DataFrame:
    
    data_url = parameters["data_location"]
    creditcard = pd.read_csv(data_url)
    
    return creditcard


def split_data(creditcard: pd.DataFrame, parameters: Dict) -> pd.DataFrame:

    seed = parameters["seed"]
    test_size = parameters["test_size"]
    train_df, holdout_df = train_test_split(creditcard, test_size=test_size, random_state=seed)
    
    return train_df, holdout_df


def run_experiment(train_df: pd.DataFrame, parameters: Dict) -> LinearRegression:

    # URL of the raw YAML file in the GitHub repository
    url = parameter["model_yaml"]

    # Send a GET request to the URL
    response = requests.get(url)

    # Raise an exception if the request was unsuccessful
    response.raise_for_status()

    # Load the YAML data from the response text
    config = yaml.safe_load(response.text)

    # Set your output directory path
    output_dir = r"C:\Users\johna\anaconda3\envs\kedro-env\kedro-project\fraud-detection\data"

    # Set up your experiment
    model = LudwigModel(config=config)
    experiment_results = model.experiment(
      dataset=train_df,
      output_directory=output_dir
    )
    
    
    
    
    
    return regressor


def evaluate_model(
    regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)