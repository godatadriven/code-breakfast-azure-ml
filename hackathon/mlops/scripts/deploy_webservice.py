import logging

# Needs to be here, as AzureML otherwise overrides our logging.
logging.basicConfig(
    format="[%(asctime)-15s] %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

from pathlib import Path

import click
from dotenv import load_dotenv

from azureml.core import Environment, Webservice, Workspace
from azureml.core.model import Model, InferenceConfig
from azureml.exceptions import WebserviceException


load_dotenv()


@click.command()
@click.option("--subscription_id", envvar="AZURE_ML_SUBSCRIPTION", required=True)
@click.option("--resource_group", envvar="AZURE_ML_RESOURCE_GROUP", required=True)
@click.option("--workspace_name", envvar="AZURE_ML_WORKSPACE", required=True)
@click.option("--model_name", required=True)
@click.option("--model_dir", required=True, type=Path)
@click.option("--script_file", default="scripts/serve.py")
@click.option("--service_name", default=None)
@click.option("--wait/--no-wait", default=True)
def main(
    subscription_id,
    resource_group,
    workspace_name,
    model_name,
    model_dir,
    script_file,
    service_name,
    wait,
):
    service_name = service_name or model_name

    logging.info("Fetching model '%s' from workspace '%s'", model_name, workspace_name)
    workspace = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
    )
    model = Model(workspace=workspace, name=model_name)

    conda_env_path = model_dir / "environment.yml"

    logging.info("Setting up inference environment")
    logging.info("  model_dir: %s", model_dir)
    logging.info("  conda_env: %s", conda_env_path)
    logging.info("  script_file: %s", model_dir / script_file)

    env = Environment.from_conda_specification(f"{model_name}-env", conda_env_path)

    inference_config = InferenceConfig(
        entry_script=script_file, source_directory=model_dir, environment=env
    )

    # Remove any existing service under the same name.
    try:
        existing_service = Webservice(workspace, name=service_name)
        logging.warning("Deleting existing service '%s'", service_name)
        existing_service.delete()
    except WebserviceException:
        pass

    logging.warning("Starting deployment")
    service = Model.deploy(workspace, service_name, [model], inference_config)

    if wait:
        service.wait_for_deployment(show_output=True)

    logging.info("Finished deploying service to %s", service.scoring_uri)


if __name__ == "__main__":
    main()
