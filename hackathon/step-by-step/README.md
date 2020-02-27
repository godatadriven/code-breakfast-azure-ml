# Azure Step-by-Step hackathon

In this hackathon, to get you started and familiar with AzureML

The goal of this part of the hackaton is to touch various components of the AzureML. Use Python in notebooks to interact with these components. We connect, train, register and deploy a model in a step by step basis. There are various challenges which you can add or implement. You do _not_ need to be a data-science to model the solution and follow these steps. 

## Getting started

To start the hackathon, follow these steps:

* Create a conda environment using:
  ```
  cd hackaton/step-by-step/
  conda env update --file environment.yml
  conda activate azureml-titanic
  ```
  Alternatively, you can create an equivalent environment using pyenv, pip, etc. if you don't have conda involved (see the environment.yml file to see which dependencies you should install).
* Start Jupyter in this environment using:
  ```
  jupyter notebook
  ```

> For using your laptop as compute target you need docker. If you do not have docker, use a remote (AMLCompute) target.

## Follow the notebooks

Open the notebooks in order, so start with `01-Connect-to-Workspace`.
