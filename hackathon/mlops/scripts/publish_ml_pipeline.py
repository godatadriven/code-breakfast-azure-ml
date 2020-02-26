#!/usr/bin/env python
# coding: utf-8

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

from azureml.core import Workspace, RunConfiguration, Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep


load_dotenv()


@click.command()
@click.option("--subscription_id", envvar="AZURE_ML_SUBSCRIPTION", required=True)
@click.option("--resource_group", envvar="AZURE_ML_RESOURCE_GROUP", required=True)
@click.option("--workspace_name", envvar="AZURE_ML_WORKSPACE", required=True)
@click.option("--pipeline_name", required=True)
@click.option("--pipeline_description", default="")
@click.option("--model_dir", required=True, type=Path)
@click.option("--cluster_name", required=True)
@click.option("--cluster_vm_size", default="STANDARD_D2_V2")
@click.option("--cluster_min_nodes", default=0, type=int)
@click.option("--cluster_idle_timeout", default="900", type=str)
@click.option(
    "--pipeline_id_path",
    default=None,
    type=Path,
    help="Output file to write the ID of the published pipeline to.",
)
def main(
    subscription_id,
    resource_group,
    workspace_name,
    pipeline_name,
    pipeline_description,
    model_dir,
    cluster_name,
    cluster_vm_size,
    cluster_min_nodes,
    cluster_idle_timeout,
    pipeline_id_path,
):
    workspace = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
    )

    target = _get_or_create_cluster(
        workspace,
        name=cluster_name,
        vm_size=cluster_vm_size,
        min_nodes=cluster_min_nodes,
        idle_seconds_before_scaledown=cluster_idle_timeout,
    )

    run_config = RunConfiguration()
    run_config.environment = Environment.from_conda_specification(
        "model-env", model_dir / "environment.yml"
    )

    datastore = workspace.get_default_datastore()

    train_data = DataReference(
        datastore=datastore,
        data_reference_name="train_data",
        path_on_datastore="titanic",
    )
    preprocessed_data = PipelineData("preprocessed_data", datastore=datastore)
    model_data = PipelineData(
        "model_data", datastore=datastore, pipeline_output_name="model"
    )

    preprocess_step = PythonScriptStep(
        name="preprocess",
        script_name="scripts/preprocess.py",
        arguments=["--input_dir", train_data, "--output_dir", preprocessed_data],
        inputs=[train_data],
        outputs=[preprocessed_data],
        compute_target=target,
        runconfig=run_config,
        source_directory=str(model_dir),
        allow_reuse=False,
    )

    train_step = PythonScriptStep(
        name="train",
        script_name="scripts/train.py",
        arguments=["--input_dir", preprocessed_data, "--model_dir", model_data],
        inputs=[preprocessed_data],
        outputs=[model_data],
        compute_target=target,
        runconfig=run_config,
        source_directory=str(model_dir),
        allow_reuse=False,
    )

    evaluate_step = PythonScriptStep(
        name="evaluate",
        script_name="scripts/evaluate.py",
        arguments=["--input_dir", preprocessed_data, "--model_dir", model_data],
        inputs=[preprocessed_data, model_data],
        compute_target=target,
        runconfig=run_config,
        source_directory=str(model_dir),
        allow_reuse=False,
    )

    register_step = PythonScriptStep(
        name="register",
        script_name="scripts/register.py",
        arguments=["--model_dir", model_data, "--model_name", "titanic"],
        inputs=[model_data],
        compute_target=target,
        runconfig=run_config,
        source_directory=str(model_dir),
        allow_reuse=False,
    )

    register_step.run_after(evaluate_step)

    pipeline = Pipeline(
        workspace=workspace,
        steps=[preprocess_step, train_step, evaluate_step, register_step],
    )
    pipeline.validate()

    published_pipeline = pipeline.publish(
        name=pipeline_name,
        description=pipeline_description,
        continue_on_step_failure=False,
    )

    if pipeline_id_path:
        with pipeline_id_path.open("w") as file_:
            print(published_pipeline.id, file=file_)


def _get_or_create_cluster(
    workspace,
    name,
    vm_size="STANDARD_D2_V2",
    vm_priority="lowpriority",
    min_nodes=0,
    max_nodes=4,
    idle_seconds_before_scaledown="300",
    wait=False,
):
    logger = logging.getLogger(__name__)

    try:
        target = ComputeTarget(workspace=workspace, name=name)
        logger.info("Using existing cluster %s", name)
    except ComputeTargetException:
        logger.info("Creating cluster %s", name)
        config = AmlCompute.provisioning_configuration(
            vm_size=vm_size,
            vm_priority=vm_priority,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            idle_seconds_before_scaledown=idle_seconds_before_scaledown,
        )
        target = ComputeTarget.create(workspace, name, config)

        if wait:
            target.wait_for_completion(show_output=False)

    return target


if __name__ == "__main__":
    main()
