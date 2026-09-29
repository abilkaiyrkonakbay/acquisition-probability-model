from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted

from acq_probability.config import SplitConfig
from acq_probability.features import (
    BASELINE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    model_matrix,
)
from acq_probability.metrics import calculate_metrics

TARGET = "acquired_within_12m"


@dataclass(frozen=True)
class TemporalDataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass
class ModelResult:
    model: BaseEstimator
    model_name: str
    candidate_metrics: pd.DataFrame
    test_metrics: dict
    backtest: pd.DataFrame
    test_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    protocol: dict


class PlattCalibratedClassifier(ClassifierMixin, BaseEstimator):
    """Calibrate a *prefitted, frozen* base model on a separate natural-rate sample.

    fit() fits only the calibration layer. It never refits the base estimator.
    A negative fitted slope is replaced by a constant empirical-prior forecast,
    so weak calibration evidence cannot silently reverse the learned ranking.
    """

    def __init__(self, base_model: Pipeline):
        self.base_model = base_model

    def _log_odds(self, x: pd.DataFrame) -> np.ndarray:
        probability = np.clip(self.base_model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(probability / (1 - probability)).reshape(-1, 1)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> PlattCalibratedClassifier:
        if y.nunique() != 2:
            raise ValueError("Calibration requires both classes; use a larger cross-section")
        self.calibrator_ = LogisticRegression(C=1_000, max_iter=1_000)
        self.calibrator_.fit(self._log_odds(x), y)
        self.constant_fallback_ = bool(self.calibrator_.coef_[0, 0] < 0)
        if self.constant_fallback_:
            self.calibrator_.coef_[:] = 0
            rate = float(y.mean())
            self.calibrator_.intercept_[:] = np.log(rate / (1 - rate))
        self.classes_ = np.array([0, 1])
        self.feature_names_in_ = self.base_model.feature_names_in_
        self.n_features_in_ = self.base_model.n_features_in_
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "calibrator_")
        return self.calibrator_.predict_proba(self._log_odds(x))

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def temporal_split(frame: pd.DataFrame, config: SplitConfig) -> TemporalDataSplit:
    """Window allocation; outcome-maturity purging happens at every fit origin."""
    dates = pd.to_datetime(frame["observation_date"])
    train = frame.loc[dates <= config.train_end].copy()
    validation = frame.loc[(dates > config.train_end) & (dates <= config.validation_end)].copy()
    test = frame.loc[(dates > config.validation_end) & (dates <= config.test_end)].copy()
    for name, partition in {"train": train, "validation": validation, "test": test}.items():
        if partition.empty:
            raise ValueError(f"Temporal {name} partition is empty; check split dates")
    return TemporalDataSplit(train=train, validation=validation, test=test)


def mature_history(frame: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """End-of-day convention: only fully observed horizons ending <= origin."""
    end = frame["observation_date"] + pd.DateOffset(months=12)
    return frame.loc[(frame["observation_date"] < origin) & (end <= origin)].copy()


def _candidate_specs() -> dict[str, tuple[str, list[str]]]:
    no_momentum = [
        f
        for f in NUMERIC_FEATURES
        if f not in {"relative_return_12m", "discount_x_underperformance"}
    ]
    return {
        "baseline_logistic": ("logistic", BASELINE_NUMERIC_FEATURES),
        "research_logistic": ("logistic", NUMERIC_FEATURES),
        "research_boosting": ("boosting", NUMERIC_FEATURES),
        "research_without_underperformance": ("logistic", no_momentum),
    }


def _fit_pipeline(name: str, frame: pd.DataFrame, random_seed: int) -> Pipeline:
    family, numeric_features = _candidate_specs()[name]
    numeric = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    if family == "boosting":
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=random_seed,
        )
    else:
        classifier = LogisticRegression(
            C=0.45,
            class_weight="balanced",
            max_iter=2_000,
            random_state=random_seed,
        )
    x, y = model_matrix(frame)
    if y.nunique() != 2:
        raise ValueError("Mature training history requires both target classes")
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    kwargs = {}
    if family == "boosting":
        kwargs["classifier__sample_weight"] = compute_sample_weight("balanced", y=y)
    pipeline.fit(x, y, **kwargs)
    return pipeline


def train_and_evaluate(
    split: TemporalDataSplit,
    *,
    top_fraction: float = 0.10,
    random_seed: int = 42,
) -> ModelResult:
    """Walk-forward family selection -> dedicated calibration -> frozen test.

    Neither calibration outcomes nor any test outcomes select a candidate.
    All selection outcomes must already be mature at the calibration origin.
    """
    history = pd.concat([split.train, split.validation], ignore_index=True)
    calibration_date = split.validation["observation_date"].max()
    selection = split.validation.loc[
        (split.validation["observation_date"] < calibration_date)
        & (split.validation["observation_date"] + pd.DateOffset(months=12) <= calibration_date)
    ]
    fold_dates = sorted(selection["observation_date"].unique())
    if not fold_dates:
        raise ValueError("Need selection periods with mature outcomes before the calibration date")
    calibration = split.validation.loc[split.validation["observation_date"] == calibration_date]
    first_test = split.test["observation_date"].min()
    if calibration_date + pd.DateOffset(months=12) > first_test:
        raise ValueError(
            "Calibration outcomes overlap the first test date; leave a full 12-month gap "
            "between the calibration observation and the first test observation"
        )

    candidate_rows, fold_rows = [], []
    for name in _candidate_specs():
        oos = []
        for fold_date in fold_dates:
            origin = pd.Timestamp(fold_date)
            train = mature_history(history, origin)
            validation = selection.loc[selection["observation_date"] == origin]
            pipeline = _fit_pipeline(name, train, random_seed)
            x, y = model_matrix(validation)
            probability = pipeline.predict_proba(x)[:, 1]
            metrics, _ = calculate_metrics(
                y, probability, validation["observation_date"], top_fraction=top_fraction
            )
            fold_rows.append(
                {
                    "model": name,
                    "fold_date": origin,
                    "train_rows": len(train),
                    "latest_train_label_end": (
                        train["observation_date"] + pd.DateOffset(months=12)
                    ).max(),
                    **metrics,
                }
            )
            oos.append(pd.DataFrame({"actual": y.to_numpy(), "score": probability, "date": origin}))
        pooled = pd.concat(oos, ignore_index=True)
        metrics, _ = calculate_metrics(
            pooled["actual"], pooled["score"], pooled["date"], top_fraction=top_fraction
        )
        candidate_rows.append({"model": name, "selection_folds": len(fold_dates), **metrics})
    candidate_metrics = (
        pd.DataFrame(candidate_rows)
        .sort_values(
            ["top_decile_lift", "average_precision"],
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )
    if candidate_metrics["top_decile_lift"].isna().all():
        raise ValueError("No events in selection folds; expand the development sample")
    selected_name = str(candidate_metrics.iloc[0]["model"])
    development = mature_history(history, calibration_date)
    base_model = _fit_pipeline(selected_name, development, random_seed)
    x_calibration, y_calibration = model_matrix(calibration)
    selected_model = PlattCalibratedClassifier(base_model).fit(x_calibration, y_calibration)

    x_test, y_test = model_matrix(split.test)
    probability = selected_model.predict_proba(x_test)[:, 1]
    test_metrics, backtest = calculate_metrics(
        y_test, probability, split.test["observation_date"], top_fraction=top_fraction
    )
    predictions = split.test[
        ["company_id", "company_name", "observation_date", "sector", TARGET]
    ].copy()
    predictions["predicted_probability"] = probability
    protocol = {
        "selection_dates": [pd.Timestamp(d).date().isoformat() for d in fold_dates],
        "selection_folds": len(fold_dates),
        "selection_warning": "Only one selection fold" if len(fold_dates) == 1 else None,
        "calibration_date": calibration_date.date().isoformat(),
        "calibration_label_end": (calibration_date + pd.DateOffset(months=12)).date().isoformat(),
        "calibration_rows": len(calibration),
        "calibration_events": int(y_calibration.sum()),
        "calibration_base_rate": float(y_calibration.mean()),
        "calibration_constant_fallback": selected_model.constant_fallback_,
        "development_rows": len(development),
        "latest_development_label_end": (development["observation_date"] + pd.DateOffset(months=12))
        .max()
        .date()
        .isoformat(),
        "first_test_date": first_test.date().isoformat(),
        "test_model_policy": "frozen; no test-period refit or selection",
        "bootstrap_unit": "company history; conditional on observed test dates",
    }
    return ModelResult(
        selected_model,
        selected_name,
        candidate_metrics,
        test_metrics,
        backtest,
        predictions,
        pd.DataFrame(fold_rows),
        protocol,
    )
