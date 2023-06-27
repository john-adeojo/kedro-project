"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import read_data, split_data, run_experiment, register_model_artefacts, run_predictions, model_training_diagnostics 


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=read_data,
                inputs=None,
                outputs="crditcard"
                name='Import Data'
            ),
            node(
                func=split_data,
                inputs="creditcard",
                outputs=["train_df", "holdout_df"],
                name="Split Data"
            ),
            node(
                func=run_experiment,
                inputs="train_df",
                outputs=None,
                name="Run Experiment"
            ),
            node(
                func=register_model_artefacts,
                inputs=None,
                outputs=None,
                name="Register Model Artefacts"
            ),
            node(
                func=run_predictions,
                inputs="holdout_df",
                outputs="full_predictions",
                name="Run Predictions"
            ),
            node(
                func=model_training_diagnostics,
                inputs="full_predictions",
                outputs=["loss_plot", "roc_curve"],
                name="Model Diagnostics"
            )
        ]
    )
