import sys
from os import path

sys.path.append(path.join(path.dirname(__file__), ".."))

from pathlib import Path

import click
import pandas as pd

from azureml.core import Run

from titanic.model import Model


@click.command()
@click.option("--input_dir", required=True, type=Path)
@click.option("--input_name", default="train.csv")
@click.option("--model_dir", required=True, type=Path)
@click.option("--model_file", default="model.joblib")
def main(input_dir, input_name, model_dir, model_file):
    run = Run.get_context()

    dataset = pd.read_csv(input_dir / input_name)

    X = dataset.drop("Survived", axis=1)
    y = dataset["Survived"]

    model = Model.load(model_dir / model_file)
    metrics = model.evaluate(X, y=y)

    # Try to get parent (for pipelines), else use run itself.
    parent_run = run.parent if hasattr(run, "parent") else run
    print(run)
    print(run.parent)
    print(run.parent.parent)
    print(parent_run)

    for param_name, param_value in model.get_params().items():
        parent_run.log(param_name, param_value)

    for metric_name, metric_value in metrics.items():
        parent_run.log(metric_name, metric_value)


if __name__ == "__main__":
    main()
