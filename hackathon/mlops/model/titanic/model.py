# -*- coding: utf-8 -*-

"""Classes for creating and persisting (fitted) ML models."""

import joblib

from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import ColumnSelector


class Model:
    """Base class representing a basic model.

    This class defines a generic interface for a machine learning model, which consists
    of a single `fit` method. Implementations of `fit` should take care of fitting
    the model on a specific dataset and return a `ModelFit` instance, which represents
    a read-only model fit that can be used to perform predictions and can be persisted.
    """

    def get_params(self):
        """Returns parameters used for fitting the model."""
        return {}

    def fit(self, X, y):
        """
        Fits model on given dataset.

        Parameters
        ----------
        X : pd.Dataframe
            Dataframe containing training data (features only, no response).
        y : Union[pd.Series, np.ndarray]
            Pandas series (or numpy array) containing the response values for the
            given training dataset.

        Returns
        -------
        Model
            Returns the model itself, after fitting.
        """
        return self

    def predict(self, X):
        """
        Produces predictions for the given dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to produce predictions for.
        """
        raise NotImplementedError()

    @classmethod
    def evaluate(self, X, y):
        """
        Evaluates the model fit on a (validation/test) dataset.

        Parameters
        ----------
        X : pd.Dataframe
            Dataframe containing the dataset feature values (no response).
        y : Union[pd.Series, np.ndarray]
            Pandas series (or numpy array) containing the response values for the
            given dataset.
        """

    @classmethod
    def load(cls, file_path):
        """Loads model fit from given file path.

        Parameters
        ----------
        file_path : str
            Path to a pickled model file.

        Returns
        -------
        ModelFit
            The unpickled model instance.

        """
        return joblib.load(file_path)

    def save(self, file_path):
        """Saves model fit to given file path.

        Parameters
        ----------
        file_path : str
            Path to save the pickled model to.

        """
        joblib.dump(self, file_path)


class NotFitError(Exception):
    """Exception indicating that the corresponding model has not been fit."""


class TitanicModel(Model):
    """A RandomForest-based model for predicting survival in the Titanic dataset."""

    def __init__(self, n_trees=200):
        super().__init__()
        self._n_trees = n_trees
        self._estimator = None

    def get_params(self):
        return {"n_trees": self._n_trees}

    def fit(self, X, y):
        self._estimator = self._build_estimator()
        self._estimator.fit(X[["Pclass", "Sex"]], y=y)
        return self

    def evaluate(self, X, y):
        if not self._estimator:
            raise ValueError("Model has not been fit")

        scorer = metrics.make_scorer(metrics.mean_squared_error)
        cv_results = cross_validate(self._estimator, X=X, y=y, scoring=scorer, cv=5)

        return {"mse": cv_results["test_score"].mean()}

    def _build_estimator(self):
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "passenger_class",
                    SimpleImputer(strategy="most_frequent"),
                    ["Pclass"],
                ),
                (
                    "sex",
                    Pipeline(
                        steps=[
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(drop="first")),
                        ]
                    ),
                    ["Sex"],
                ),
            ],
            remainder="drop",
        )

        pipeline = Pipeline(
            steps=[
                ("select_columns", ColumnSelector(columns=["Pclass", "Sex"])),
                ("preprocessing", preprocessor),
                ("model", RandomForestClassifier(n_estimators=self._n_trees)),
            ]
        )

        return pipeline

    def predict(self, X):
        if self._estimator is None:
            raise NotFitError("Model has not yet been fit")
        return self._estimator.predict(X[["Pclass", "Sex"]])
