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
import pandas as pd
import requests

from azureml.core import Environment, Webservice, Workspace
from azureml.core.model import Model, InferenceConfig
from azureml.exceptions import WebserviceException


load_dotenv()


@click.command()
@click.option("--subscription_id", envvar="AZURE_ML_SUBSCRIPTION", required=True)
@click.option("--resource_group", envvar="AZURE_ML_RESOURCE_GROUP", required=True)
@click.option("--workspace_name", envvar="AZURE_ML_WORKSPACE", required=True)
@click.option("--service_name", required=True)
@click.option("--test_dir", required=True, type=Path)
@click.option("--test_file", default="test.csv")
def main(
    subscription_id, resource_group, workspace_name, service_name, test_dir, test_file
):
    workspace = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
    )

    # pylint: disable=abstract-class-instantiated
    service = Webservice(workspace, name=service_name)

    test_path = test_dir / test_file
    test_df = pd.read_csv(test_path)

    response = requests.post(
        service.scoring_uri,
        data=test_df.to_json(),
        headers={"Content-type": "application/json"},
    )
    response.raise_for_status()

    print(response.json())


if __name__ == "__main__":
    main()
