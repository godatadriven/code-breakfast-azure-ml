import json
from sklearn.externals import joblib
from azureml.core.model import Model
import pandas as pd


def init():
    global model
    # note here "titanic-sklearn-dctree" is the name of the model registered under the workspace
    # this call should return the path to the model.pkl file on the local disk.
    model_path = Model.get_model_path(model_name='titanic-sklearn-dctree')
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)


# note you can pass in multiple rows for scoring
def run(raw_data):
    try:
        print("DEBUGGING: Printing RAW DATA")
        print(raw_data)
        input_data = json.loads(raw_data)['data']
        df = pd.DataFrame.from_records(data=input_data,
                                       columns=['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
                                                'Ticket', 'Fare', 'Cabin', 'Embarked'])
        result = model.predict(df)

        # you can return any data type as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        result = "ERROR: " + str(e)
        return result
