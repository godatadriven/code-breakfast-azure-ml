import sys
from os import path

sys.path.append(path.join(path.dirname(__file__), ".."))

from pathlib import Path

import click
import pandas as pd

from titanic.model import TitanicModel


@click.command()
@click.option("--input_dir", required=True, type=Path)
@click.option("--input_file", default="train.csv")
@click.option("--model_dir", default="outputs", type=Path)
@click.option("--model_file", default="model.joblib")
@click.option("--n_trees", default=200, type=int)
def main(input_dir, input_file, model_dir, model_file, n_trees):
    # Fetch our dataset.
    dataset = pd.read_csv(input_dir / input_file)

    X = dataset.drop("Survived", axis=1)
    y = dataset["Survived"]

    # Train our model.
    model = TitanicModel(n_trees=n_trees)
    model.fit(X, y=y)

    # Save our model output.
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / model_file)


if __name__ == "__main__":
    main()
