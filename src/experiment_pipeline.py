"""Class-based, leakage-aware experiment orchestration for churn modeling."""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .config import MODEL_DIR, OUTPUT_DIR, PROCESSED_DATA_DIR, RANDOM_STATE
from .data_loader import load_raw_data, split_features_target
from .data_validation import audit_dataframe, validate_frame
from .feature_engineering import FeatureEngineer
from .feature_selection import FeatureSelector
from .threshold_optimization import threshold_table, select_cost_threshold
from .utils import save_json

LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetSplits:
    ids_train: pd.Series
    ids_validation: pd.Series
    ids_test: pd.Series
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


class ChurnExperimentPipeline:
    """Run leakage-aware CatBoost feature-set experiments without touching test data."""

    def __init__(self, random_state: int = RANDOM_STATE, artifact_root: Path | None = None) -> None:
        self.random_state = random_state
        self.artifact_root = Path(artifact_root or MODEL_DIR).resolve()
        self.output_root = OUTPUT_DIR.resolve()
        self.splits: DatasetSplits | None = None
        self.experiment_results: list[dict[str, Any]] = []

    def load_and_split(self, path=None) -> DatasetSplits:
        frame = load_raw_data(path) if path is not None else load_raw_data()
        validate_frame(frame)
        audit = audit_dataframe(frame)
        self.output_root.joinpath("tables").mkdir(parents=True, exist_ok=True)
        save_json(audit, self.output_root / "tables" / "data_audit.json")
        ids, raw_features, target = split_features_target(frame)
        development, test, y_development, y_test, ids_development, ids_test = train_test_split(raw_features, target, ids, test_size=0.2, stratify=target, random_state=self.random_state)
        train, validation, y_train, y_validation, ids_train, ids_validation = train_test_split(development, y_development, ids_development, test_size=0.2, stratify=y_development, random_state=self.random_state)
        self.splits = DatasetSplits(ids_train, ids_validation, ids_test, train, validation, test, y_train, y_validation, y_test)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        for name, values in {
            "X_train": train, "X_validation": validation, "X_test": test,
            "y_train": y_train, "y_validation": y_validation, "y_test": y_test,
        }.items():
            values.to_csv(PROCESSED_DATA_DIR / f"{name}.csv", index=False)
        return self.splits

    def feature_sets(self) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        if self.splits is None:
            raise RuntimeError("Call load_and_split before feature_sets")
        engineer = FeatureEngineer()
        monthly_charge_threshold = self.splits.X_train["MonthlyCharges"].median()
        train = engineer.transform(self.splits.X_train, monthly_charge_threshold)
        validation = engineer.transform(self.splits.X_validation, monthly_charge_threshold)
        test = engineer.transform(self.splits.X_test, monthly_charge_threshold)
        selector = FeatureSelector()
        return {
            "raw": (self.splits.X_train, self.splits.X_validation, self.splits.X_test),
            "engineered": (train, validation, test),
            "selected": (selector.select(train), selector.select(validation), selector.select(test)),
        }

    def _prepare_xgboost(self, train: pd.DataFrame, *frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
        """One-hot encode categories and impute values using training-only statistics."""
        categorical = train.select_dtypes(exclude="number").columns.tolist()
        numeric = train.select_dtypes(include="number").columns.tolist()
        numeric_medians = train[numeric].median()
        category_modes = train[categorical].mode(dropna=True).iloc[0] if categorical else pd.Series(dtype=object)

        def prepare(frame: pd.DataFrame) -> pd.DataFrame:
            prepared = frame.copy()
            for column in numeric:
                prepared[column] = prepared[column].fillna(numeric_medians[column])
            for column in categorical:
                prepared[column] = prepared[column].fillna(category_modes[column]).astype(str)
            return pd.get_dummies(prepared, columns=categorical, dtype=float)

        prepared_train = prepare(train)
        prepared_frames = [prepared_train]
        for frame in frames:
            prepared_frames.append(prepare(frame).reindex(columns=prepared_train.columns, fill_value=0.0))
        return tuple(prepared_frames)

    def _tune_xgboost(
        self,
        train: pd.DataFrame,
        y_train: pd.Series,
        validation: pd.DataFrame,
        y_validation: pd.Series,
    ) -> XGBClassifier:
        """Tune a bounded XGBoost candidate set on the validation split."""
        candidates = [
            {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.08, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 1},
            {"n_estimators": 120, "max_depth": 4, "learning_rate": 0.06, "subsample": 0.90, "colsample_bytree": 0.90, "min_child_weight": 3},
        ]
        best_model: XGBClassifier | None = None
        best_score = -np.inf
        for candidate in candidates:
            model = XGBClassifier(**candidate, eval_metric="logloss", random_state=self.random_state, n_jobs=-1)
            model.fit(train, y_train)
            score = average_precision_score(y_validation, model.predict_proba(validation)[:, 1])
            LOGGER.info("XGBoost candidate %s validation PR-AUC: %.4f", candidate, score)
            if score > best_score:
                best_model = model
                best_score = score
        assert best_model is not None
        return best_model

    def _prepare_catboost(self, train: pd.DataFrame, *frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
        """Learn fill values on training data and apply them to all experiment splits."""
        categorical = train.select_dtypes(exclude="number").columns.tolist()
        numeric = train.select_dtypes(include="number").columns.tolist()
        medians = train[numeric].median()

        def prepare(frame: pd.DataFrame) -> pd.DataFrame:
            prepared = frame.copy()
            for column in categorical:
                prepared[column] = prepared[column].fillna("Missing").astype(str)
            for column in numeric:
                prepared[column] = prepared[column].fillna(medians[column])
            return prepared

        return tuple(prepare(frame) for frame in (train, *frames)) + (categorical,)

    def _tune_catboost(
        self,
        train: pd.DataFrame,
        y_train: pd.Series,
        validation: pd.DataFrame,
        y_validation: pd.Series,
        categorical: list[str],
    ) -> CatBoostClassifier:
        """Tune a bounded, reproducible CatBoost candidate set on development data."""
        candidates = [
            {"iterations": 180, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3.0, "random_strength": 0.5, "bagging_temperature": 0.0},
            {"iterations": 260, "depth": 5, "learning_rate": 0.04, "l2_leaf_reg": 6.0, "random_strength": 1.0, "bagging_temperature": 0.5},
        ]
        best_model: CatBoostClassifier | None = None
        best_score = -np.inf
        for candidate in candidates:
            model = CatBoostClassifier(
                **candidate,
                eval_metric="PRAUC",
                loss_function="Logloss",
                verbose=50,
                random_seed=self.random_state,
                thread_count=-1,
                allow_writing_files=False,
            )
            model.fit(
                train,
                y_train,
                cat_features=categorical,
                eval_set=(validation, y_validation),
                early_stopping_rounds=30,
                verbose=50,
            )
            score = average_precision_score(y_validation, model.predict_proba(validation)[:, 1])
            LOGGER.info("CatBoost candidate %s validation PR-AUC: %.4f", candidate, score)
            if score > best_score:
                best_model = model
                best_score = score
        assert best_model is not None
        return best_model

    def _save_roc_curve(self, y: pd.Series, probabilities: np.ndarray, path: Path, label: str) -> None:
        false_positive_rate, true_positive_rate, _ = roc_curve(y, probabilities)
        plt.figure(figsize=(6, 5))
        plt.plot(false_positive_rate, true_positive_rate, label=f"{label} (ROC-AUC = {roc_auc_score(y, probabilities):.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray", label="No-skill")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"{label} ROC curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=160)
        plt.close()

    def _prepare_legacy_catboost(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Compatibility helper retained for older notebook imports."""
        prepared = frame.copy()
        categorical = prepared.select_dtypes(exclude="number").columns.tolist()
        for column in categorical:
            prepared[column] = prepared[column].fillna("Missing").astype(str)
        for column in prepared.select_dtypes(include="number"):
            prepared[column] = prepared[column].fillna(prepared[column].median())
        return prepared, categorical

    def _evaluate(self, y: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
        predicted = (probabilities >= threshold).astype(int)
        return {"threshold": threshold, "roc_auc": float(roc_auc_score(y, probabilities)), "pr_auc": float(average_precision_score(y, probabilities)), "precision": float(precision_score(y, predicted, zero_division=0)), "recall": float(recall_score(y, predicted, zero_division=0)), "f1": float(f1_score(y, predicted, zero_division=0))}

    def run_single(self, feature_name: str, model_name: str) -> dict[str, Any]:
        """Tune, evaluate, and persist one model-feature representation experiment."""
        if self.splits is None:
            self.load_and_split()
        assert self.splits is not None
        train, validation, test = self.feature_sets()[feature_name]
        LOGGER.info("Tuning %s on %s features", model_name, feature_name)
        if model_name == "catboost":
            train_model, validation_model, test_model, categorical_columns = self._prepare_catboost(train, validation, test)
            model = self._tune_catboost(train_model, self.splits.y_train, validation_model, self.splits.y_validation, categorical_columns)
        elif model_name == "xgboost":
            train_model, validation_model, test_model = self._prepare_xgboost(train, validation, test)
            categorical_columns = []
            model = self._tune_xgboost(train_model, self.splits.y_train, validation_model, self.splits.y_validation)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        validation_probability = model.predict_proba(validation_model)[:, 1]
        test_probability = model.predict_proba(test_model)[:, 1]
        thresholds = threshold_table(self.splits.y_validation, validation_probability, cost_fn=2.0, cost_fp=1.0)
        threshold = select_cost_threshold(thresholds)
        result = {
            "feature_set": feature_name,
            "model": model_name,
            "validation": self._evaluate(self.splits.y_validation, validation_probability, threshold),
            "test": self._evaluate(self.splits.y_test, test_probability, threshold),
            "threshold": threshold,
        }
        artifact_dir = self.artifact_root / f"{feature_name}_{model_name}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, artifact_dir / "model.joblib")
        if model_name == "catboost":
            model.save_model(str(artifact_dir / "model.cbm"))
        save_json(result, artifact_dir / "metrics.json")
        save_json({"categorical_features": categorical_columns, "feature_columns": train.columns.tolist(), "best_hyperparameters": model.get_params()}, artifact_dir / "metadata.json")
        pd.DataFrame({"customerID": self.splits.ids_test, "actual_churn": self.splits.y_test, "churn_probability": test_probability, "predicted_churn": (test_probability >= threshold).astype(int)}).to_csv(self.output_root / "predictions" / f"{feature_name}_{model_name}_test.csv", index=False)
        self._save_roc_curve(self.splits.y_test, test_probability, self.output_root / "figures" / f"{feature_name}_{model_name}_roc_auc.png", f"{feature_name} {model_name}")
        return result

    def finalize_artifacts(self) -> pd.DataFrame:
        """Create the comparison table and copy the best persisted experiment."""
        records = []
        valid_experiment_names = {
            f"{feature_set}_{model_name}"
            for feature_set in ("raw", "engineered", "selected")
            for model_name in ("catboost", "xgboost")
        }
        for experiment_name in valid_experiment_names:
            metrics_file = self.artifact_root / experiment_name / "metrics.json"
            if not metrics_file.exists():
                continue
            records.append(json.loads(metrics_file.read_text(encoding="utf-8")))
        if not records:
            raise RuntimeError("No completed experiment artifacts found")
        results = pd.json_normalize(records, sep="_").sort_values(["validation_pr_auc", "validation_f1"], ascending=False)
        self.output_root.joinpath("tables").mkdir(parents=True, exist_ok=True)
        results.to_csv(self.output_root / "tables" / "experiment_comparison.csv", index=False)
        best = results.iloc[0].to_dict()
        save_json(best, self.artifact_root / "best_model.json")
        best_artifact = self.artifact_root / f"{best['feature_set']}_{best['model']}"
        best_root = self.artifact_root / "best_model"
        if best_root.exists():
            shutil.rmtree(best_root)
        shutil.copytree(best_artifact, best_root)
        shutil.copy2(self.output_root / "figures" / f"{best['feature_set']}_{best['model']}_roc_auc.png", best_root / "roc_auc.png")
        return results

    def run(self) -> pd.DataFrame:
        if self.splits is None:
            self.load_and_split()
        assert self.splits is not None
        self.experiment_results = []
        for feature_name, (train, validation, test) in self.feature_sets().items():
            for model_name in ("catboost", "xgboost"):
                self.experiment_results.append(self.run_single(feature_name, model_name))
        return self.finalize_artifacts()
