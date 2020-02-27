import os

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from azureml.core import Run

run = Run.get_context()

# Here we reference in input dataset by name:
titanic_ds = run.input_datasets["titanic_training"]
# validation = run.input_datasets["validation"]

df: pd.DataFrame = titanic_ds.to_pandas_dataframe()
# test_ds: pd.DataFrame = validation.to_pandas_dataframe()

print(df)

# Model, here the data scientist operates :)
# ------------------------------------------------

X_train = df.drop("Survived", axis=1)
y_train = df["Survived"]

cat_preprocess = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder()
)

ct = make_column_transformer(
    (SimpleImputer(strategy="median"), ["Pclass", "Age", "SibSp", "Parch", "Fare"]),
    (cat_preprocess, ["Cabin", "Sex"])
)

pipeline = Pipeline([
    ('preprocess', ct),
    ('classifier', DecisionTreeClassifier()),
])

clf = pipeline.fit(X_train, y_train)

# Challenge: log model metrics:

# X_test = test_ds.drop("Survived", axis=1)
# y_test = test_ds["Survived"]
# predictions = clf.predict(X_test)
# mse = mean_squared_error(y_test, predictions)
# use 'mse' to log as a metric, in this run.


# Writing OUTPUT
# ------------------------------------------------
model_file_name = 'decision_tree.pkl'

# Create SKLearn fitted model in 'magic' outputs folder:
os.makedirs('./outputs', exist_ok=True)

with open(model_file_name, 'wb') as file:
    joblib.dump(value=clf, filename='outputs/' + model_file_name)
