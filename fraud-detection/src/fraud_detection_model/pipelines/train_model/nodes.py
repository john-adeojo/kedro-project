"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.10
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_score, recall_score, f1_score, auc
import matplotlib
import matplotlib.pyplot as plt
from ludwig.api import LudwigModel
import requests
import yaml
import json
from kedro.framework.session import KedroSession
#from kedro.framework.context import load_context
from kedro.config import ConfigLoader
from kedro.io import DataCatalog
from pathlib import Path
from typing import Dict, Any, Tuple
import plotly.graph_objects as go


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


def read_data(parameters: Dict[str, Any]) -> pd.DataFrame:
    
    data_url = parameters["data_location"]
    creditcard = pd.read_csv(data_url)
    
    return creditcard


def split_data(creditcard: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:

    seed = parameters["seed"]
    test_size = parameters["test_size"]
    train_df, holdout_df = train_test_split(creditcard, test_size=test_size, random_state=seed)
    
    return train_df, holdout_df


def run_experiment(train_df: pd.DataFrame, parameters: Dict[str, Any]) -> None:

    # URL of the raw YAML file in the GitHub repository
    url = parameters["model_yaml"]

    # Send a GET request to the URL
    response = requests.get(url)

    # Raise an exception if the request was unsuccessful
    response.raise_for_status()

    # Load the YAML data from the response text
    config = yaml.safe_load(response.text)

    # Set your output directory path
    output_dir = parameters["output_dir"]

    # Set up your experiment
    model = LudwigModel(config=config)
    experiment_results = model.experiment(
      dataset=train_df,
      output_directory=output_dir
    )
    
    return None


def register_model_artefacts(parameters: Dict[str, Any]) -> None:
    
    # create the holdout_predictions directory
    predictions_dir = str(Path(latest_experiment_dir).parent / 'holdout_predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    # Load the Kedro context
    #context = load_context()
    
    session = KedroSession.create("fraud_detection_model")
    context = session.load_context()

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
        'roc_curve': {
            'type': 'kedro.extras.datasets.matplotlib.MatplotlibWriter',
            'filepath': str(Path(latest_experiment_dir) / 'roc_curve.png'),
        },
        'loss_plot': {
            'type': 'kedro.extras.datasets.plotly.PlotlyDataSet',
            'filepath': str(Path(latest_experiment_dir) / 'loss_plot.html'),
        },
    }

    # Load the catalog configuration
    config_loader = ConfigLoader('conf/base')
    catalog = DataCatalog.from_config(catalog_config, config_loader)

    # Add the catalog to the context
    context.catalog = catalog

    return None

def run_predictions(parameters: Dict[str, Any], holdout_df) -> pd.DataFrame:
    
    # Load the Kedro context
    # context = load_context()
    session = KedroSession.create("fraud_detection_model")
    context = session.load_context()

    # Load the model weights
    model_weights = context.catalog.load('model')

    # Load the model
    model = LudwigModel.load(model_weights)
    
    # run predictions on holdout
    predictions, _ = model.predict(dataset=holdout_df)
    
    full_predictions = predictions.merge(right=holdout_df,   left_index=True, right_index=True)
    full_predictions['Class_predictions'] = full_predictions['Class_predictions'].map({True: 1, False: 0})
    
    return full_predictions


def model_training_diagnostics(parameters: Dict[str, Any], full_predictions) -> Tuple[matplotlib.figure.Figure, go.Figure]:
    
    # plot roc curve 
    fpr, tpr, thresholds = roc_curve(full_predictions['Class'], full_predictions['Class_predictions'])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    roc_curve = plt.figure()
    
    # plot loss curve

    # Get the latest experiment directory
    output_dir = parameters["output_dir"]
    latest_experiment_dir = get_latest_experiment_dir(output_dir)

    json_path = latest_experiment_dir + "/training_statistics.json"

    # Load the JSON file
    with open(json_path, 'r') as f:
        train_stats = json.load(f)

        train_loss = train_stats['training']['Class']['loss']
        validation_loss = train_stats['validation']['Class']['loss']
        test_loss = train_stats['test']['Class']['loss']

    # Create list of epochs
    epochs = list(range(1, len(train_loss) + 1))

    # Create the plot
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', name='Training loss'))
    fig.add_trace(go.Scatter(x=epochs, y=validation_loss, mode='lines', name='Validation loss'))
    fig.add_trace(go.Scatter(x=epochs, y=test_loss, mode='lines', name='Test loss'))

    # Add details
    fig.update_layout(title='Training, Validation and Test Loss', xaxis_title='Epochs', yaxis_title='Loss')
    
    loss_plot = fig
    
    return loss_plot, roc_curve
    

    
    

    
    