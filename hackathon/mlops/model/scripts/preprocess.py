import sys
from os import path

sys.path.append(path.join(path.dirname(__file__), ".."))

from pathlib import Path

import click
import pandas as pd


@click.command()
@click.option("--input_dir", required=True, type=Path)
@click.option("--input_file", default="train.csv")
@click.option("--output_dir", required=True, type=Path)
def main(input_dir, input_file, output_dir):
    data = pd.read_csv(input_dir / input_file)
    subset = data[["Pclass", "Sex", "Survived"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_dir / input_file, index=False)


if __name__ == "__main__":
    main()
