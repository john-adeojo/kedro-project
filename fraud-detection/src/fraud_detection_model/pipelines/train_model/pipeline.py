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
                inputs="params:data_location",
                outputs="creditcard",
                name='Import_Data'
            ),
            node(
                func=split_data,
                inputs=["creditcard", "params:model_options"],
                outputs=["train_df", "holdout_df"],
                name="Split_Data"
            ),
            node(
                func=run_experiment,
                inputs=["train_df", "params:model_yaml", "params:output_dir"],
                outputs="exp_run",
                name="Run_Experiment"
            ),
            node(
                func=register_model_artefacts,
                inputs=["exp_run", "params:output_dir"],
                outputs="register_model",
                name="Register_Model_Artefacts"
            ),
            node(
                func=run_predictions,
                inputs=["holdout_df", "register_model"],
                outputs="full_predictions",
                name="Run_Predictions"
            ),
            node(
                func=model_training_diagnostics,
                inputs=["full_predictions", "params:output_dir"],
                outputs=["loss_plot", "roc_curve"],
                name="Model_Diagnostics"
            )
        ]
    )
