# Azure MLOps hackathon

In this hackathon, we'll start exploring doing MLOps in Azure ML.

The mean idea behind the hackathon is to learn how to set up a Machine Learning Pipeline in Azure ML and to see how we can use such a pipeline to train + deploy new versions of our models in Azure ML. Besides this, we'll also set up a CI/CD pipeline Azure DevOps, which will allow us to automate the process of retraining + re-deploying our model + model pipeline whenever we make changes to our code.

## Getting started

To start the hackathon, follow these steps:

* Create a conda environment using:
  ```
  conda env create --file environment.yml
  conda activate azureml-titanic-mlops
  ```
  Alternatively, you can create an equivalent environment using pyenv, pip, etc. if you don't have conda involved (see the environment.yml file to see which dependencies you should install).
* In parallel, make sure to download the `config.json` file for your Azure ML workspace from the Azure Portal and save it in the project directory. (You can find this file by opening the ML resource in the Azure portal and clicking on `'Download config.json'`, under overview).
* Start Jupyter lab in this environment using:
  ```
  jupyter lab
  ```

## Exercises

Open the first notebook in the `notebooks` directory and proceed through the exercises.
