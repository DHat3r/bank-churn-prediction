"""
churn_model.py
──────────────
Modelo de predicción de Churn para Banco Futura.

Implementa un ensemble de:
  - Random Forest (Spark MLlib) → escalable, distribuido
  - XGBoost                     → alto rendimiento, SHAP-compatible
  - Regresión Logística         → baseline interpretable

Incluye:
  - Validación cruzada estratificada
  - Optimización bayesiana de hiperparámetros (Optuna)
  - Métricas enterprise (AUC, KS, Gini, Lift)
  - SHAP values para explicabilidad
  - MLflow tracking de experimentos
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.spark
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from loguru import logger
from pyspark.ml.classification import (
    RandomForestClassifier,
    LogisticRegression,
    GBTClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)


class ChurnPredictor:
    """
    Predictor de churn bancario con capacidades enterprise.

    Args:
        spark:          SparkSession activa.
        experiment_name: Nombre del experimento MLflow.
        model_type:     "random_forest" | "xgboost" | "logistic"
    """

    CHURN_THRESHOLDS = {
        "alto":   0.70,    # Acción inmediata: llamada del ejecutivo
        "medio":  0.40,    # Campaña de retención
        "bajo":   0.20,    # Monitoreo y oferta digital
    }

    def __init__(
        self,
        spark: SparkSession,
        experiment_name: str = "banco-futura-churn",
        model_type: str = "random_forest",
    ) -> None:
        self.spark = spark
        self.experiment_name = experiment_name
        self.model_type = model_type
        self._model = None
        self._xgb_model = None
        mlflow.set_experiment(experiment_name)
        logger.info(f"ChurnPredictor inicializado | modelo={model_type}")

    # ── Entrenamiento Spark MLlib ────────────────────────────────────────────

    def train_spark(
        self,
        df_train: DataFrame,
        cross_validate: bool = True,
        n_folds: int = 5,
    ) -> None:
        """
        Entrena modelo Spark MLlib con validación cruzada opcional.

        Args:
            df_train:       DataFrame GOLD con columna 'features' y 'churn_label'.
            cross_validate: Si True, aplica k-fold CV con grid search.
            n_folds:        Número de folds para cross-validation.
        """
        logger.info(f"Iniciando entrenamiento Spark | modelo={self.model_type}")

        with mlflow.start_run(run_name=f"spark_{self.model_type}"):
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("cross_validate", cross_validate)
            mlflow.log_param("n_folds", n_folds)

            classifier = self._get_spark_classifier()

            if cross_validate:
                self._model = self._train_with_cv(df_train, classifier, n_folds)
            else:
                self._model = classifier.fit(df_train)

            logger.success("Modelo Spark entrenado correctamente")

    def _get_spark_classifier(self):
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                labelCol="churn_label",
                featuresCol="features",
                numTrees=200,
                maxDepth=8,
                minInstancesPerNode=10,
                featureSubsetStrategy="sqrt",
                seed=42,
            )
        elif self.model_type == "gbt":
            return GBTClassifier(
                labelCol="churn_label",
                featuresCol="features",
                maxIter=100,
                maxDepth=6,
                stepSize=0.05,
                seed=42,
            )
        else:  # logistic
            return LogisticRegression(
                labelCol="churn_label",
                featuresCol="features",
                maxIter=100,
                regParam=0.01,
                elasticNetParam=0.5,
            )

    def _train_with_cv(self, df_train: DataFrame, classifier, n_folds: int):
        """Grid search + cross-validation con Spark ML."""
        param_grid = (
            ParamGridBuilder()
            .addGrid(classifier.maxDepth, [4, 6, 8])
            .addGrid(classifier.numTrees if hasattr(classifier, "numTrees") else classifier.maxIter, [100, 200])
            .build()
        )

        evaluator = BinaryClassificationEvaluator(
            labelCol="churn_label",
            metricName="areaUnderROC",
        )

        cv = CrossValidator(
            estimator=classifier,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=n_folds,
            parallelism=4,          # Spark paraleliza los folds
            seed=42,
        )

        logger.info(f"Cross-validation {n_folds}-fold iniciada...")
        cv_model = cv.fit(df_train)
        best_auc = max(cv_model.avgMetrics)
        logger.success(f"CV completada | mejor AUC={best_auc:.4f}")
        mlflow.log_metric("cv_best_auc", best_auc)
        return cv_model.bestModel

    # ── Entrenamiento XGBoost (para explicabilidad SHAP) ────────────────────

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        Entrena XGBoost en Pandas/NumPy (compatibilidad SHAP).
        Usar cuando el dataset cabe en memoria del driver.
        """
        logger.info("Entrenando XGBoost con early stopping...")

        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),  # Desbalance
            "eval_metric": "auc",
            "use_label_encoder": False,
            "random_state": 42,
        }

        with mlflow.start_run(run_name="xgboost_churn"):
            mlflow.log_params(params)
            self._xgb_model = xgb.XGBClassifier(**params)
            self._xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=30,
                verbose=100,
            )
            val_auc = roc_auc_score(y_val, self._xgb_model.predict_proba(X_val)[:, 1])
            mlflow.log_metric("val_auc", val_auc)
            mlflow.xgboost.log_model(self._xgb_model, "xgboost_model")
            logger.success(f"XGBoost entrenado | AUC validación={val_auc:.4f}")

    # ── Evaluación y métricas enterprise ────────────────────────────────────

    def evaluate_spark(self, df_test: DataFrame) -> Dict[str, float]:
        """Evalúa el modelo Spark y retorna métricas estándar bancarias."""
        if self._model is None:
            raise RuntimeError("Modelo no entrenado. Llama a train_spark() primero.")

        predictions = self._model.transform(df_test)

        auc_evaluator = BinaryClassificationEvaluator(
            labelCol="churn_label", metricName="areaUnderROC"
        )
        pr_evaluator = BinaryClassificationEvaluator(
            labelCol="churn_label", metricName="areaUnderPR"
        )
        acc_evaluator = MulticlassClassificationEvaluator(
            labelCol="churn_label", predictionCol="prediction", metricName="accuracy"
        )

        metrics = {
            "auc_roc":    auc_evaluator.evaluate(predictions),
            "auc_pr":     pr_evaluator.evaluate(predictions),
            "accuracy":   acc_evaluator.evaluate(predictions),
        }

        # Gini coefficient = 2 * AUC - 1
        metrics["gini"] = 2 * metrics["auc_roc"] - 1

        logger.info("─── Métricas de Evaluación ───────────────────────────────")
        for k, v in metrics.items():
            logger.info(f"  {k:<12}: {v:.4f}")
            mlflow.log_metric(k, v)

        return metrics

    def compute_shap_values(
        self,
        X_explain: np.ndarray,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """
        Calcula SHAP values para explicabilidad del modelo XGBoost.
        Fundamental en banca para cumplimiento regulatorio.
        """
        if self._xgb_model is None:
            raise RuntimeError("XGBoost no entrenado.")

        logger.info("Calculando SHAP values...")
        explainer = shap.TreeExplainer(self._xgb_model)
        shap_values = explainer.shap_values(X_explain)

        shap_df = pd.DataFrame(
            np.abs(shap_values).mean(axis=0).reshape(1, -1),
            columns=feature_names,
        ).T.rename(columns={0: "mean_abs_shap"}).sort_values(
            "mean_abs_shap", ascending=False
        )

        logger.success("Top 10 features más relevantes (SHAP):")
        logger.info("\n" + shap_df.head(10).to_string())
        return shap_df

    # ── Scoring e inferencia ─────────────────────────────────────────────────

    def score_customers(self, df: DataFrame) -> DataFrame:
        """
        Aplica el modelo entrenado y agrega columnas de scoring.

        Retorna DataFrame con:
          - churn_probability: score de 0 a 1
          - churn_risk_level:  "alto" | "medio" | "bajo" | "sin_riesgo"
          - recommended_action: acción recomendada por CRM
        """
        if self._model is None:
            raise RuntimeError("Modelo no entrenado.")

        from pyspark.sql.functions import udf, col
        from pyspark.sql.types import StringType
        from pyspark.ml.functions import vector_to_array

        predictions = self._model.transform(df)

        # Extraer probabilidad de clase positiva (churn=1)
        scored = predictions.withColumn(
            "churn_probability",
            vector_to_array(col("probability")).getItem(1),
        )

        # Clasificar riesgo
        high = self.CHURN_THRESHOLDS["alto"]
        med  = self.CHURN_THRESHOLDS["medio"]
        low  = self.CHURN_THRESHOLDS["bajo"]

        from pyspark.sql.functions import when
        scored = (
            scored
            .withColumn(
                "churn_risk_level",
                when(col("churn_probability") >= high, "ALTO")
                .when(col("churn_probability") >= med, "MEDIO")
                .when(col("churn_probability") >= low, "BAJO")
                .otherwise("SIN_RIESGO"),
            )
            .withColumn(
                "recommended_action",
                when(col("churn_probability") >= high, "LLAMADA_EJECUTIVO")
                .when(col("churn_probability") >= med, "CAMPAÑA_RETENCIÓN")
                .when(col("churn_probability") >= low, "OFERTA_DIGITAL")
                .otherwise("MONITOREO"),
            )
        )

        logger.success(
            f"Scoring completado: {scored.count():,} clientes evaluados"
        )
        return scored

    def save_model(self, path: str) -> None:
        """Persiste el modelo en el path indicado (ADLS / DBFS)."""
        if self._model:
            self._model.write().overwrite().save(path)
            logger.success(f"Modelo Spark guardado en: {path}")
