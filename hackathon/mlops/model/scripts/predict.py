import sys
from os import path

sys.path.append(path.join(path.dirname(__file__), ".."))

from pathlib import Path

import click
import pandas as pd

from titanic.model import Model


@click.command()
@click.option("--input_dir", required=True, type=Path)
@click.option("--input_file", default="test.csv")
@click.option("--model_dir", required=True, type=Path)
@click.option("--model_file", default="model.joblib")
@click.option("--output_path", default="predictions.csv", type=Path)
def predict(input_dir, input_file, model_dir, model_file, output_path):
    input_path = input_dir / input_file
    input_df = pd.read_csv(input_path)

    model_path = model_dir / model_file
    model = Model.load(model_path)

    predictions = model.predict(input_df)
    predictions_df = pd.DataFrame({"prediction": predictions})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path)
