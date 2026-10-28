"""Baselines from Section III-C on identical drug-level features.

  * sole-source rule (Eq. 7)
  * L2 logistic regression
  * gradient boosting (XGBoost if installed, else sklearn's HistGradientBoosting)
  * edge-free MLP is in `model.EdgeFreeMLP`, trained by the same loop as the GNN
"""

from __future__ import annotations

import numpy as np

from .graph import FEATURE_NAMES_DRUG


def _col(name: str) -> int:
    return FEATURE_NAMES_DRUG.index(name)


def sole_source_scores(Xd: np.ndarray) -> np.ndarray:
    """s_v = 1[n_labelers == 1] + eps * 26-week recall count (Eq. 7)."""
    n_labelers = Xd[:, _col("n_labelers")]
    recalls = Xd[:, _col("recall_I_26w")] + Xd[:, _col("recall_II_26w")] + Xd[:, _col("recall_III_26w")]
    return (n_labelers == 1).astype(float) + 1e-3 * recalls


class LogisticBaseline:
    def __init__(self, *, C: float = 1.0, horizon: int = 1):
        from sklearn.linear_model import LogisticRegression

        self.horizon = horizon
        self.model = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", solver="lbfgs")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticBaseline":
        self.model.fit(X, y[:, self.horizon])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class GradientBoostingBaseline:
    """XGBoost when available, otherwise sklearn's histogram boosting."""

    def __init__(self, *, horizon: int = 1, random_state: int = 0):
        self.horizon = horizon
        try:
            from xgboost import XGBClassifier

            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=random_state,
            )
            self.backend = "xgboost"
        except Exception:  # pragma: no cover - depends on the environment
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.model = HistGradientBoostingClassifier(random_state=random_state)
            self.backend = "sklearn_hist"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingBaseline":
        self.model.fit(X, y[:, self.horizon])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
