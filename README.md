# code-breakfast-azure-ml

Material for the AzureML code breakfast.

## Hackathons

The material contains two different hackathons for exploring Azure ML.

### Step-by-step intro to Azure ML

The first hackathon, step-by-step focusses on giving a step-by-step introduction
to Azure ML, in which we step through the different components of Azure ML using
a play-through example in notebooks.

You can find the material for this hackathon under `hackathon/step-by-step`.

### MLOps in Azure ML

In the second hackathon, we provide a small example of how to get started with
Azure ML Pipelines and how to use these pipelines to do MLOps in Azure ML. The
hackathon starts off with a Notebook example of how to start building an ML pipeline
and how to trigger the pipeline to train a model. In the final part of the hackathon,
we have an open-ended exercise aimed at automating the entire CI/CD process of your
model using Azure Pipelines.

You can find the material for this hackathon under `hackathon/mlops`.

## Extra challenges

Should you manage to finish both hackathons, here are some extra ideas you can
try implementing in either example case:

- Upload a File and register as a Tabular DataSet
- Connect a second StorageAccount (gen2) as a DataStore
- Connect a SqlDatabase as a DataStore
- Train/register a your model as an SkLearn Model (after fit)
- Create a new version of a data set with more records
- Change a data set and use the data drift detection
- Use Databricks as a compute target for a large datasets
- Monitoring: collect metrics from a running model
