from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLUMNS = [
    "station",
    "sim_day",
    "doy",
    "topwt",
    "grnwt",
    "xlai",
    "totir",
    "tofer",
    "swfac",
    "nstres",
]
TARGET_COLUMNS = ["expert_action_irrigation", "expert_action_n"]
NUMERIC_FEATURE_COLUMNS = [col for col in FEATURE_COLUMNS if col != "station"]


def postprocess_actions(prediction, threshold: float = 25.0, max_irrigation: float = 100.0, max_n: float = 150.0) -> np.ndarray:
    arr = np.asarray(prediction, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    arr[:, 0] = np.where(arr[:, 0] >= threshold, arr[:, 0], 0.0)
    arr[:, 1] = np.where(arr[:, 1] >= threshold, arr[:, 1], 0.0)
    arr[:, 0] = np.clip(arr[:, 0], 0.0, max_irrigation)
    arr[:, 1] = np.clip(arr[:, 1], 0.0, max_n)
    return arr


class FeatureFrameCaster(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        if isinstance(x, pd.DataFrame):
            return x[FEATURE_COLUMNS].copy()
        return pd.DataFrame(x, columns=FEATURE_COLUMNS)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("station", OneHotEncoder(handle_unknown="ignore"), ["station"]),
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                NUMERIC_FEATURE_COLUMNS,
            ),
        ],
        remainder="drop",
    )


def make_random_forest_regressor(random_state: int = 0) -> Pipeline:
    return Pipeline(
        [
            ("cast", FeatureFrameCaster()),
            ("preprocess", make_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_mlp_regressor(random_state: int = 0) -> Pipeline:
    return Pipeline(
        [
            ("cast", FeatureFrameCaster()),
            ("preprocess", make_preprocessor()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=1000,
                    early_stopping=True,
                    random_state=random_state,
                ),
            ),
        ]
    )


@dataclass
class ConstantScheduleBaseline:
    schedule_table: pd.DataFrame

    def predict(self, x, deterministic: bool = True):
        frame = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x, columns=FEATURE_COLUMNS)
        actions = []
        for _, row in frame.iterrows():
            station = str(row["station"])
            sim_day = int(round(float(row["sim_day"])))
            match = self.schedule_table[
                (self.schedule_table["station"].astype(str) == station)
                & (self.schedule_table["sim_day"].astype(int) == sim_day)
            ]
            if len(match):
                actions.append([float(match["expert_action_irrigation"].iloc[0]), float(match["expert_action_n"].iloc[0])])
            else:
                actions.append([0.0, 0.0])
        return np.asarray(actions, dtype=float)


@dataclass
class TwoStageClassifierRegressor:
    threshold: float = 25.0
    random_state: int = 0

    def __post_init__(self):
        self.irrigation_classifier = None
        self.n_classifier = None
        self.irrigation_regressor = None
        self.n_regressor = None
        self.irrigation_has_positive = False
        self.n_has_positive = False

    def _classifier(self):
        return Pipeline(
            [
                ("cast", FeatureFrameCaster()),
                ("preprocess", make_preprocessor()),
                ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=1, random_state=self.random_state, n_jobs=-1)),
            ]
        )

    def _regressor(self):
        return Pipeline(
            [
                ("cast", FeatureFrameCaster()),
                ("preprocess", make_preprocessor()),
                ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=1, random_state=self.random_state, n_jobs=-1)),
            ]
        )

    def fit(self, x: pd.DataFrame, y):
        target = pd.DataFrame(y, columns=TARGET_COLUMNS)
        irrigation_event = target["expert_action_irrigation"].astype(float) > 0
        n_event = target["expert_action_n"].astype(float) > 0
        self.irrigation_has_positive = bool(irrigation_event.any())
        self.n_has_positive = bool(n_event.any())
        if self.irrigation_has_positive:
            self.irrigation_classifier = self._classifier().fit(x, irrigation_event.astype(int))
            self.irrigation_regressor = self._regressor().fit(x.loc[irrigation_event], target.loc[irrigation_event, "expert_action_irrigation"])
        if self.n_has_positive:
            self.n_classifier = self._classifier().fit(x, n_event.astype(int))
            self.n_regressor = self._regressor().fit(x.loc[n_event], target.loc[n_event, "expert_action_n"])
        return self

    def predict(self, x: pd.DataFrame, deterministic: bool = True):
        frame = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x, columns=FEATURE_COLUMNS)
        pred = np.zeros((len(frame), 2), dtype=float)
        if self.irrigation_has_positive and self.irrigation_classifier is not None and self.irrigation_regressor is not None:
            event = self.irrigation_classifier.predict(frame).astype(bool)
            if event.any():
                pred[event, 0] = self.irrigation_regressor.predict(frame.loc[event])
        if self.n_has_positive and self.n_classifier is not None and self.n_regressor is not None:
            event = self.n_classifier.predict(frame).astype(bool)
            if event.any():
                pred[event, 1] = self.n_regressor.predict(frame.loc[event])
        return postprocess_actions(pred, threshold=self.threshold)


def supervised_metrics(y_true, y_pred) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    metrics: dict[str, float] = {}
    labels = [("irrigation", 0), ("nitrogen", 1)]
    for label, idx in labels:
        metrics[f"{label}_mae"] = float(mean_absolute_error(truth[:, idx], pred[:, idx]))
        metrics[f"{label}_rmse"] = float(mean_squared_error(truth[:, idx], pred[:, idx]) ** 0.5)
        true_event = truth[:, idx] > 0
        pred_event = pred[:, idx] > 0
        precision, recall, f1, _ = precision_recall_fscore_support(true_event, pred_event, average="binary", zero_division=0)
        metrics[f"{label}_event_precision"] = float(precision)
        metrics[f"{label}_event_recall"] = float(recall)
        metrics[f"{label}_event_f1"] = float(f1)
    true_any = (truth > 0).any(axis=1)
    pred_any = (pred > 0).any(axis=1)
    metrics["nonzero_action_accuracy"] = float(accuracy_score(true_any, pred_any))
    zero_mask = ~true_any
    metrics["zero_action_accuracy"] = float(accuracy_score(true_any[zero_mask], pred_any[zero_mask])) if zero_mask.any() else float("nan")
    return metrics


def as_feature_row(station: str, sim_day: int, doy: int, latest: dict) -> pd.DataFrame:
    row = {
        "station": station,
        "sim_day": int(sim_day),
        "doy": int(doy),
        "topwt": latest.get("topwt", 0.0),
        "grnwt": latest.get("grnwt", 0.0),
        "xlai": latest.get("xlai", 0.0),
        "totir": latest.get("totir", 0.0),
        "tofer": latest.get("tofer", 0.0),
        "swfac": latest.get("swfac", 0.0),
        "nstres": latest.get("nstres", 0.0),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_real_action(policy, features: pd.DataFrame, postprocess: bool = True, threshold: float = 25.0):
    pred = policy.predict(features)
    if isinstance(pred, tuple):
        pred = pred[0]
    arr = np.asarray(pred, dtype=float)
    if postprocess:
        arr = postprocess_actions(arr, threshold=threshold)
    return {"amir": float(arr.reshape(-1, 2)[0, 0]), "anfer": float(arr.reshape(-1, 2)[0, 1])}
