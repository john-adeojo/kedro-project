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
                inputs="train_model",
                outputs="crditcard",
                name='Import_Data'
            ),
            node(
                func=split_data,
                inputs=["creditcard", "train_model"],
                outputs=["train_df", "holdout_df"],
                name="Split_Data"
            ),
            node(
                func=run_experiment,
                inputs=["train_df", "train_model"],
                outputs=None,
                name="Run_Experiment"
            ),
            node(
                func=register_model_artefacts,
                inputs="train_model",
                outputs=None,
                name="Register_Model_Artefacts"
            ),
            node(
                func=run_predictions,
                inputs=["holdout_df", "train_model"],
                outputs="full_predictions",
                name="Run_Predictions"
            ),
            node(
                func=model_training_diagnostics,
                inputs=["full_predictions", "train_model"],
                outputs=["loss_plot", "roc_curve"],
                name="Model_Diagnostics"
            )
        ]
    )
