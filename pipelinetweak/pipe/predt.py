import sklearn.base
import numpy as np


class PredT(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Wraps a sklearn estimator (ClusterMixin, ClassifierMixin,
    RegressorMixin) to use the output of their .predict method
    as .transform output

    Args:
        model (sklearn.base.BaseEstimator): A sklearn regression or
            classification model

    Example:
        from sklearn.pipeline import Pipeline
        from pipelinetweak.pipe import PredT
        from sklearn.linear_model import LinearRegression
        from sklearn.dummy import DummyRegressor
        model = Pipeline(steps=[
            ('trans', PredT(LinearRegression())),
            ('pred', DummyRegressor())
        ])
        model.fit(X, y)
        Y_pred = model.predict(X)
    """
    def __init__(self, model: sklearn.base.BaseEstimator) -> None:
        self.model = model

    def fit(self, *args, **kwargs) -> 'PredT':
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X: np.ndarray, **transform_params) -> np.ndarray:
        Z = self.model.predict(X)
        return Z.reshape(-1, 1) if len(Z.shape) == 1 else Z
