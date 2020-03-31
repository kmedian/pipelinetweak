import sklearn.base
import numpy as np


class ProbT(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Wraps a sklearn classifier (ClassifierMixin) to use the output
    of their .predict_proba method.

    Args:
        model (sklearn.base.ClassifierMixin): A sklearn classification model
        drop (bool): Flag if to drop the column with the 0 probabilties

    Example:
        from sklearn.pipeline import Pipeline
        from pipelinetweak.pipe import ProbT
        from sklearn.linear_model import LinearRegression
        from sklearn.dummy import DummyRegressor
        model = Pipeline(steps=[
            ('trans', ProbT(LogisticRegression(), drop=False)),
            ('pred', DummyRegressor())
        ])
        model.fit(X, y)
        Y_pred = model.predict(X)
    """
    def __init__(self,
                 model: sklearn.base.ClassifierMixin,
                 drop: bool = True) -> None:
        self.model = model
        self.drop = drop

    def fit(self, *args, **kwargs) -> 'ProbT':
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X: np.ndarray, **transform_params) -> np.ndarray:
        Z = self.model.predict_proba(X)
        if self.drop:
            return Z[:, 1:]
        else:
            return Z
