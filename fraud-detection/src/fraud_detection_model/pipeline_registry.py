"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from fraud_detection_model.pipelines import train_model as tm


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    
    train_model_pipeline = tm.create_pipeline()
    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())
    
    
    return {
        "__default__": train_model_pipeline,
        "__train_model__": train_model_pipeline
        }
