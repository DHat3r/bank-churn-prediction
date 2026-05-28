"""
feature_engineering.py
───────────────────────
Pipeline de Feature Engineering en PySpark para Banco Futura.

Transforma la capa SILVER (datos limpios) en features de ML (capa GOLD).

Patrón aplicado: Medallion Architecture
  RAW  → limpieza básica, schema enforcement
  SILVER → deduplicación, tipado, nulls
  GOLD   → features derivadas, encodings, normalizaciones

Features generadas:
  - Comportamentales:   tendencia de saldo, ratio transaccional
  - Temporales:         recencia, ventanas móviles
  - Financieras:        CLV, scoring de rentabilidad
  - Digitales:          engagement score
  - Interacción:        complaint intensity, NPS bucket

"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    StandardScaler,
    VectorAssembler,
    Imputer,
)
from loguru import logger


class BankingFeatureEngineer:
    """
    Transforma datos de clientes bancarios en features para ML.

    Args:
        spark: SparkSession activa.
    """

    # Features numéricas que entran al modelo
    NUMERIC_FEATURES = [
        "age",
        "tenure_months",
        "avg_balance_6m",
        "monthly_transactions",
        "app_logins_30d",
        "n_products",
        "clv_score",
        "n_complaints_3m",
        "nps_score",
        "days_since_last_contact",
        # Features derivadas (se agregan en transform())
        "balance_per_tenure",
        "txn_per_product",
        "digital_engagement_score",
        "complaint_intensity",
        "nps_bucket",
    ]

    CATEGORICAL_FEATURES = ["region", "segment"]

    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark
        self._pipeline: Pipeline | None = None
        self._pipeline_model = None

    # ── Capa SILVER: limpieza y tipado ──────────────────────────────────────

    def raw_to_silver(self, df: DataFrame) -> DataFrame:
        """
        Aplica reglas de calidad de datos básicas.
        Resultado: capa SILVER del Lakehouse.
        """
        logger.info("Iniciando transformación RAW → SILVER")

        df_silver = (
            df
            # Eliminar duplicados por customer_id (last write wins)
            .dropDuplicates(["customer_id"])
            # Cast explícito de tipos
            .withColumn("age", F.col("age").cast(IntegerType()))
            .withColumn("avg_balance_6m", F.col("avg_balance_6m").cast(DoubleType()))
            .withColumn("clv_score", F.col("clv_score").cast(DoubleType()))
            # Imputar nulls con valores de negocio seguros
            .withColumn("n_complaints_3m", F.coalesce(F.col("n_complaints_3m"), F.lit(0)))
            .withColumn("app_logins_30d", F.coalesce(F.col("app_logins_30d"), F.lit(0)))
            .withColumn("nps_score", F.coalesce(F.col("nps_score"), F.lit(5)))
            # Eliminar registros sin customer_id o churn_label
            .filter(F.col("customer_id").isNotNull())
            .filter(F.col("churn_label").isNotNull())
            # Agregar columna de procesamiento
            .withColumn("silver_ts", F.current_timestamp())
        )

        original_count = df.count()
        silver_count = df_silver.count()
        logger.info(
            f"SILVER: {silver_count:,} registros "
            f"({original_count - silver_count:,} eliminados por calidad)"
        )
        return df_silver

    # ── Capa GOLD: Feature Engineering ─────────────────────────────────────

    def silver_to_gold(self, df: DataFrame) -> DataFrame:
        """
        Genera todas las features derivadas.
        Resultado: capa GOLD lista para ML.
        """
        logger.info("Iniciando transformación SILVER → GOLD (Feature Engineering)")

        df_gold = (
            df
            # ── Features financieras ─────────────────────────────────
            .withColumn(
                "balance_per_tenure",
                F.when(F.col("tenure_months") > 0,
                       F.col("avg_balance_6m") / F.col("tenure_months"))
                .otherwise(0.0)
            )
            .withColumn(
                "txn_per_product",
                F.when(F.col("n_products") > 0,
                       F.col("monthly_transactions") / F.col("n_products"))
                .otherwise(0.0)
            )
            .withColumn(
                "clv_per_month",
                F.when(F.col("tenure_months") > 0,
                       F.col("clv_score") / F.col("tenure_months"))
                .otherwise(0.0)
            )
            # ── Engagement digital ────────────────────────────────────
            # Score compuesto: app_logins normalizado + transacciones normalizadas
            .withColumn(
                "digital_engagement_score",
                (F.col("app_logins_30d") * 0.6 + F.col("monthly_transactions") * 0.4)
            )
            # ── Señales de riesgo churn ───────────────────────────────
            .withColumn(
                "complaint_intensity",
                F.when(F.col("tenure_months") > 0,
                       F.col("n_complaints_3m") / (F.col("tenure_months") / 12.0))
                .otherwise(F.col("n_complaints_3m").cast(DoubleType()))
            )
            .withColumn(
                "nps_bucket",
                F.when(F.col("nps_score") <= 6, F.lit(0))    # Detractor
                .when(F.col("nps_score") <= 8, F.lit(1))     # Pasivo
                .otherwise(F.lit(2))                          # Promotor
                .cast(IntegerType())
            )
            .withColumn(
                "recency_risk",
                F.when(F.col("days_since_last_contact") > 90, F.lit(3))
                .when(F.col("days_since_last_contact") > 30, F.lit(2))
                .when(F.col("days_since_last_contact") > 14, F.lit(1))
                .otherwise(F.lit(0))
                .cast(IntegerType())
            )
            # ── Flag VIP compuesto ────────────────────────────────────
            .withColumn(
                "is_high_value",
                (
                    (F.col("avg_balance_6m") > 5_000_000) |
                    (F.col("n_products") >= 4) |
                    (F.col("clv_score") > 1_000_000)
                ).cast(IntegerType())
            )
            # Metadata
            .withColumn("gold_ts", F.current_timestamp())
        )

        logger.success(f"GOLD listo: {df_gold.count():,} registros con features enriquecidas")
        return df_gold

    # ── Pipeline ML: encoding + scaling + assembler ──────────────────────────

    def build_ml_pipeline(self) -> Pipeline:
        """
        Construye el pipeline Spark ML para preparar features al modelo.

        Etapas:
          1. StringIndexer → codifica categorías a índices numéricos
          2. OneHotEncoder → one-hot encoding de categorías
          3. VectorAssembler → combina todas las features en un vector
          4. StandardScaler → normalización Z-score

        Returns:
            Pipeline Spark ML (sin ajustar, requiere .fit())
        """
        logger.info("Construyendo pipeline Spark ML...")

        # Indexar features categóricas
        indexers = [
            StringIndexer(
                inputCol=col,
                outputCol=f"{col}_idx",
                handleInvalid="keep",
            )
            for col in self.CATEGORICAL_FEATURES
        ]

        # One-Hot Encoding
        encoders = [
            OneHotEncoder(
                inputCol=f"{col}_idx",
                outputCol=f"{col}_ohe",
                dropLast=True,
            )
            for col in self.CATEGORICAL_FEATURES
        ]

        # Imputar nulls en numéricas con mediana
        imputer = Imputer(
            inputCols=self.NUMERIC_FEATURES,
            outputCols=[f"{c}_imp" for c in self.NUMERIC_FEATURES],
            strategy="median",
        )

        # Ensamblar vector de features
        feature_cols = (
            [f"{c}_imp" for c in self.NUMERIC_FEATURES] +
            [f"{col}_ohe" for col in self.CATEGORICAL_FEATURES]
        )
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features_raw",
            handleInvalid="skip",
        )

        # Normalizar
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withMean=True,
            withStd=True,
        )

        self._pipeline = Pipeline(stages=indexers + encoders + [imputer, assembler, scaler])
        logger.success("Pipeline ML construido con éxito")
        return self._pipeline

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """Ajusta el pipeline y transforma el dataset completo."""
        if self._pipeline is None:
            self.build_ml_pipeline()
        logger.info("Ajustando pipeline ML...")
        self._pipeline_model = self._pipeline.fit(df)
        logger.success("Pipeline ajustado. Aplicando transformación...")
        return self._pipeline_model.transform(df)

    def transform(self, df: DataFrame) -> DataFrame:
        """Aplica el pipeline ya ajustado (para inferencia en producción)."""
        if self._pipeline_model is None:
            raise RuntimeError(
                "Pipeline no ajustado. Llama a fit_transform() primero."
            )
        return self._pipeline_model.transform(df)
