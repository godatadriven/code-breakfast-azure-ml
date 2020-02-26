from pathlib import Path

import click

from azureml.core import Run

from titanic.model import Model


@click.command()
@click.option("--model_dir", required=True, type=Path)
@click.option("--model_file", default="model.joblib")
@click.option("--model_name", required=True)
def main(model_dir, model_file, model_name):
    run = Run.get_context()
    parent_run = run.parent if hasattr(run, "parent") else run

    model_local_path = model_dir / model_file
    model_run_path = "./outputs/" + model_name

    parent_run.upload_file(name=model_run_path, path_or_stream=str(model_local_path))
    parent_run.register_model(model_name="titanic", model_path=model_run_path)


if __name__ == "__main__":
    main()
