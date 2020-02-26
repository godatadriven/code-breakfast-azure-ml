import os
import json

import pandas as pd

# from inference_schema.schema_decorators import input_schema, output_schema
# from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

from titanic.model import Model

model = None

# The init() method is called once, when the web service starts up.
#
# Typically you would deserialize the model file, as shown here using joblib,
# and store it in a global variable so your run() method can access it later.
def init():
    global model
    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    model_path = os.environ["AZUREML_MODEL_DIR"] + "/titanic"
    model = Model.load(model_path)


# The run() method is called each time a request is made to the scoring API.
#
# Shown here are the optional input_schema and output_schema decorators
# from the inference-schema pip package. Using these decorators on your
# run() method parses and validates the incoming payload against
# the example input you provide here. This will also generate a Swagger
# API document for your web service.
# @input_schema('data', NumpyParameterType(np.array([[0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0]])))
# @output_schema(NumpyParameterType(np.array([4429.929236457418])))
def run(data):
    predict_df = pd.read_json(data)
    predictions = model.predict(predict_df)
    return {"predictions": predictions.tolist()}


if __name__ == "__main__":
    raise RuntimeError(
        "This script is not meant to be called directly, "
        "but by the AzureML deployment runtime."
    )
