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
                inputs="parameters",
                outputs="creditcard",
                name='Import_Data'
            ),
            node(
                func=split_data,
                inputs=["creditcard", "parameters"],
                outputs=["train_df", "holdout_df"],
                name="Split_Data"
            ),
            node(
                func=run_experiment,
                inputs=["train_df", "parameters"],
                outputs="exp_run",
                name="Run_Experiment"
            ),
            node(
                func=register_model_artefacts,
                inputs=["parameters","exp_run"],
                outputs="register_model",
                name="Register_Model_Artefacts"
            ),
            node(
                func=run_predictions,
                inputs=["holdout_df"],
                outputs="full_predictions",
                name="Run_Predictions"
            ),
            node(
                func=model_training_diagnostics,
                inputs=["full_predictions", "parameters"],
                outputs=["loss_plot", "roc_curve"],
                name="Model_Diagnostics"
            )
        ]
    )
