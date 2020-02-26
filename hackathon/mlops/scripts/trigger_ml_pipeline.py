#!/usr/bin/env python
# coding: utf-8

import logging

# Needs to be here, as AzureML otherwise overrides our logging.
logging.basicConfig(
    format="[%(asctime)-15s] %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

import time

import click
from dotenv import load_dotenv
import requests

from azureml.core import Workspace
from azureml.pipeline.core import PipelineRun, PublishedPipeline
from azureml.core.authentication import InteractiveLoginAuthentication

load_dotenv()

JOB_STATUS = {
    "not_started": {0, "NotStarted"},
    "running": {1, "Running"},
    "failed": {2, "Failed"},
    "cancelled": {3, "Cancelled"},
    "finished": {4, "Finished"},
}


@click.command()
@click.option("--subscription_id", envvar="AZURE_ML_SUBSCRIPTION", required=True)
@click.option("--resource_group", envvar="AZURE_ML_RESOURCE_GROUP", required=True)
@click.option("--workspace_name", envvar="AZURE_ML_WORKSPACE", required=True)
@click.option("--pipeline_id", required=True)
@click.option("--experiment_name", required=True)
def main(subscription_id, resource_group, workspace_name, pipeline_id, experiment_name):
    workspace = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
    )

    pipeline = PublishedPipeline.get(workspace=workspace, id=pipeline_id)

    auth = InteractiveLoginAuthentication()
    aad_token = auth.get_authentication_header()

    logging.info("Rest endpoint: %s", pipeline.endpoint)
    logging.info("Submitting job...")

    request_payload = {
        "ExperimentName": experiment_name,
        "ParameterAssignments": {},
    }

    response = requests.post(pipeline.endpoint, headers=aad_token, json=request_payload)
    response.raise_for_status()

    run_id = response.json()["Id"]
    logging.info("Job ID: %s", run_id)

    pipeline_run = PipelineRun.get(workspace, run_id)
    _wait_for_completion(pipeline_run)


def _wait_for_completion(pipeline_run):
    logging.info("Waiting for job to start...")
    status = pipeline_run.get_status()
    while status in JOB_STATUS["not_started"]:
        time.sleep(1)
        status = pipeline_run.get_status()

    if status in JOB_STATUS["running"]:
        logging.info("Job started, waiting for completion...")
        while status in JOB_STATUS["running"]:
            time.sleep(1)
            status = pipeline_run.get_status()

    if status in JOB_STATUS["finished"]:
        logging.info("Job finished successfully!")
    elif status in JOB_STATUS["failed"]:
        logging.error("Job failed!")
    elif status in JOB_STATUS["cancelled"]:
        logging.warning("Job was cancelled.")
    else:
        raise ValueError(f"Unexpected status '{status}'")


if __name__ == "__main__":
    main()
