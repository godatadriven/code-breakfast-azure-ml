import pandas as pd
import requests

service = Webservice(workspace, name=service_name)

test_df = pd.read_csv("../../../datasets/predict.csv")

response = requests.post(
    service.scoring_uri,
    data=test_df.to_json(),
    headers={"Content-type": "application/json"},
)
response.raise_for_status()

print(response.json())
