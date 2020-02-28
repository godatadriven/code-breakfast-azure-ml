# Azure Step-by-Step hackathon

In this hackathon, we aim to get you familiar with the different parts of Azure ML.

The goal of this hackathon is to touch various components of the Azure ML, using Python in notebooks to explore these components. The overall idea is to connect, train, register and deploy a model in a step-by-step basis. There are various challenges which you can add or implement.  Note that you do _not_ need to be a data scientist to model the solution and follow these steps.

### Getting started on Azure Notebook VM

Working on the remove VM is preferred, because its working out of the box. No need to configure conda and setup docker.

- Launch the AzureML workspace
- Click on  the Compute tab from the menu 
- Spin up a Compute Notebook VM and wait ~5 minutes...
- Open Jupyter or JupyterLab
- Open a new Terminal
- `cd Users/your-user`
- `git clone https://github.com/godatadriven/code-breakfast-azure-ml.git codebreakfast`

The Notebook VM has the correct packages in the conda environment already present.
The `config.json` is also present on your VM, so you can skip the first step in the first notebook 

### Getting started on local machine

To start the hackathon, follow these steps:

* Make sure you have docker up 'n running
* Create a conda environment using:
  ```
  cd hackathon/step-by-step/
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

Open the notebooks in order, so start with:
 
 - `01-Connect-to-Workspace`.
 - `02-Create-Experiment`.
 - and so on...
