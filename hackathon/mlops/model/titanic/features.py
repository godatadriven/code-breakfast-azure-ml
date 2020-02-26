import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects columns from a pandas DataFrame."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X is not a dataframe")

        try:
            return X[self.columns]
        except KeyError:
            missing_columns = list(set(self.columns) - set(X.columns))
            raise KeyError(
                "The DataFrame does not include the columns: %s" % missing_columns
            )
