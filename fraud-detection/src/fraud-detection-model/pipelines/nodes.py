import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_score, recall_score, f1_score, auc
import matplotlib.pyplot as plt
from ludwig.api import LudwigModel
import requests
import yaml
import json
from kedro.framework.context import load_context
from kedro.config import ConfigLoader
from kedro.io import DataCatalog
from pathlib import Path


# Helper functions

def get_latest_experiment_dir(base_dir):
        # Get a list of all items in the base directory
        items = os.listdir(base_dir)

        # Filter out items that are not directories
        dirs = [item for item in items if os.path.isdir(os.path.join(base_dir, item))]

        # Filter out directories that do not start with 'experiment_run'
        experiment_dirs = [dir for dir in dirs if dir.startswith('experiment_run')]

        # Sort the directories by the experiment number (the part after 'experiment_run')
        sorted_experiment_dirs = sorted(experiment_dirs, key=lambda dir: int(dir.split('_')[-1]) if dir.split('_')[-1].isdigit() else -1)

        # Return the last directory in the sorted list
        latest_dir = sorted_experiment_dirs[-1] if sorted_experiment_dirs else None

        # Return the full path of the latest directory
        return os.path.join(base_dir, latest_dir).replace('\\', '/') if latest_dir else None

    
# Nodes


def read_data(parameters: Dict) -> pd.DataFrame:
    
    data_url = parameters["data_location"]
    creditcard = pd.read_csv(data_url)
    
    return creditcard


def split_data(creditcard: pd.DataFrame, parameters: Dict) -> pd.DataFrame:

    seed = parameters["seed"]
    test_size = parameters["test_size"]
    train_df, holdout_df = train_test_split(creditcard, test_size=test_size, random_state=seed)
    
    return train_df, holdout_df


def run_experiment(train_df: pd.DataFrame, parameters: Dict)

    # URL of the raw YAML file in the GitHub repository
    url = parameter["model_yaml"]

    # Send a GET request to the URL
    response = requests.get(url)

    # Raise an exception if the request was unsuccessful
    response.raise_for_status()

    # Load the YAML data from the response text
    config = yaml.safe_load(response.text)

    # Set your output directory path
    output_dir = parameter["output_dir"]

    # Set up your experiment
    model = LudwigModel(config=config)
    experiment_results = model.experiment(
      dataset=train_df,
      output_directory=output_dir
    )
    
    return None


def register_model_artefacts(parameters: Dict):
    
    # create the holdout_predictions directory
    predictions_dir = str(Path(latest_experiment_dir).parent / 'holdout_predictions')

    # Load the Kedro context
    context = load_context()

    output_dir = parameters["output_dir"]

    # Get the latest experiment directory
    latest_experiment_dir = get_latest_experiment_dir(output_dir)

    # Define the catalog configuration
    catalog_config = {
        'ludwig_model': {
            'type': 'kedro.io.PartitionedDataSet',
            'path': str(Path(latest_experiment_dir) / 'model'),
            'dataset': {
                'type': 'kedro.extras.datasets.pickle.PickleDataSet',
            },
        },
        'holdout_predictions': {
            'type': 'kedro.extras.datasets.pandas.CSVDataSet',
            'filepath': str(Path(latest_experiment_dir) / 'holdout_predictions' / 'predictions.csv'),
        },
    }

    # Load the catalog configuration
    config_loader = ConfigLoader('conf/base')
    catalog = DataCatalog.from_config(catalog_config, config_loader)

    # Add the catalog to the context
    context.catalog = catalog

    return None

def run_predictions(parameters: Dict, holdout_df):
    
    # Load the Kedro context
    context = load_context()

    # Load the model weights
    model_weights = context.catalog.load('model')

    # Load the model
    model = LudwigModel.load(model_weights)
    
    # run predictions on holdout
    predictions, _ = model.predict(dataset=holdout_df)
    
    model_analysis_df = predictions.merge(right=holdout_df,   left_index=True, right_index=True)
    model_analysis_df['Class_predictions'] = model_analysis_df['Class_predictions'].map({True: 1, False: 0})
    
    
    
    

    
    