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
from kedro.config import ConfigLoader
from kedro.io import DataCatalog
from pathlib import Path
from typing import Dict, Any, Tuple
import plotly.graph_objects as go
import os
import shutil
import glob


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
        
        # If latest_dir is None, raise an error
        if latest_dir is None:
            raise ValueError(f"No directories starting with 'experiment_run' found in {base_dir}")
        
        # Return the full path of the latest directory
        return os.path.join(base_dir, latest_dir).replace('\\', '/') if latest_dir else None
    
import glob

def delete_file():
    # Get a list of all HDF5 files in the current directory
    hdf5_files = glob.glob('*.hdf5')

    # Print each HDF5 file name
    for file in hdf5_files:
        os.remove(file)

    # Get a list of all JSON files in the current directory
    json_files = glob.glob('*.json')

    # Print each JSON file name
    for file in json_files:
        os.remove(file)


# Nodes

def read_data(data_location) -> pd.DataFrame:
    
    creditcard = pd.read_csv(data_location)
    
    return creditcard


def split_data(creditcard: pd.DataFrame, model_options) -> Tuple[pd.DataFrame, pd.DataFrame]:

    seed = model_options["seed"]
    test_size = model_options["test_size"]
    train_df, holdout_df = train_test_split(creditcard, test_size=test_size, random_state=seed)
    
    return train_df, holdout_df


def run_experiment(train_df: pd.DataFrame, model_yaml, output_dir) -> pd.DataFrame:

    # Send a GET request to the URL
    response = requests.get(model_yaml)

    # Raise an exception if the request was unsuccessful
    response.raise_for_status()

    # Load the YAML data from the response text
    config = yaml.safe_load(response.text)

    # Set up your experiment
    model = LudwigModel(config=config)
    experiment_results = model.experiment(
      dataset=train_df,
      output_directory=output_dir
    )
    
    df = pd.DataFrame()
    
    # create dummy output
    exp_run = pd.DataFrame(columns=['action'])
    
    delete_file()

    
    return exp_run


def run_predictions(holdout_df: pd.DataFrame, exp_run: pd.DataFrame, output_dir) -> pd.DataFrame:
    
    # dummpy input varibale
    df = exp_run
    
    latest_experiment_dir = get_latest_experiment_dir(output_dir)
    model_path = Path(latest_experiment_dir) / 'model'

    # Load the model
    model = LudwigModel.load(model_path)
    
    # run predictions on holdout
    predictions, _ = model.predict(dataset=holdout_df)
    
    full_predictions = predictions.merge(right=holdout_df,   left_index=True, right_index=True)
    full_predictions['Class_predictions'] = full_predictions['Class_predictions'].map({True: 1, False: 0})
    
    return full_predictions


def model_training_diagnostics(full_predictions: pd.DataFrame, output_dir) -> Tuple[matplotlib.figure.Figure, go.Figure]:
    
    # plot roc curve 
    
    # plot roc curve 
    fpr, tpr, thresholds = roc_curve(full_predictions['Class'], full_predictions['Class_predictions'])
    roc_auc = auc(fpr, tpr)

    # Create the base figure
    fig = go.Figure()

    # Add the ROC curve
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))

    # Add the random guess line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))

    # Update the layout
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=700,
        title='Receiver Operating Characteristic'
    )

    roc_curve_plot = fig
    
#     fpr, tpr, thresholds = roc_curve(full_predictions['Class'], full_predictions['Class_predictions'])
#     roc_auc = auc(fpr, tpr)

#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc="lower right")
#     roc_curve_plot = plt.gcf()
    
    # plot loss curve
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
    
    return loss_plot, roc_curve_plot
