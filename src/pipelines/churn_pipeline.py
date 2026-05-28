"""
churn_pipeline.py
──────────────────
Orquestador del pipeline end-to-end de Churn y VIP Detection.

Coordina:
  1. Ingesta de datos (RAW)
  2. Transformación SILVER (calidad)
  3. Feature Engineering GOLD
  4. Integración de embeddings BERT
  5. Entrenamiento del modelo
  6. Scoring masivo de clientes
  7. Exportación de resultados (Delta Lake / Parquet para Power BI)

Diseñado para ejecutarse en Azure Databricks como Job orquestado.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.features.feature_engineering import BankingFeatureEngineer
from src.models.churn_model import ChurnPredictor
from src.embeddings.bert_embeddings import BERTEmbeddingEngine
from src.ingestion.synthetic_data_generator import BankingDataGenerator
from src.utils.spark_session import SparkSessionFactory


class ChurnVIPPipeline:
    """
    Pipeline maestro de Churn y VIP Detection para Banco Futura.

    Args:
        env:            Entorno de ejecución (local | databricks).
        data_path:      Ruta base de datos en ADLS / DBFS.
        model_type:     Tipo de modelo Spark ML a usar.
        run_embeddings: Si True, integra BERT al pipeline.
    """

    def __init__(
        self,
        env: str = "local",
        data_path: str = "data",
        model_type: str = "random_forest",
        run_embeddings: bool = True,
    ) -> None:
        self.env = env
        self.data_path = Path(data_path)
        self.model_type = model_type
        self.run_embeddings = run_embeddings

        # Componentes del pipeline
        self.spark: SparkSession = SparkSessionFactory.get_session(env=env)
        self.feature_engineer = BankingFeatureEngineer(self.spark)
        self.churn_predictor = ChurnPredictor(
            self.spark, model_type=model_type
        )
        self.bert_engine = BERTEmbeddingEngine() if run_embeddings else None

        logger.info(
            f"ChurnVIPPipeline iniciado | env={env} | modelo={model_type} "
            f"| embeddings={'ON' if run_embeddings else 'OFF'}"
        )

    # ── Ejecución completa del pipeline ─────────────────────────────────────

    def run(self, n_customers: int = 50_000) -> DataFrame:
        """
        Ejecuta el pipeline completo de punta a punta.

        Returns:
            DataFrame con scores de churn y clasificación VIP.
        """
        logger.info("═" * 60)
        logger.info("  BANCO FUTURA | CHURN + VIP PIPELINE INICIADO")
        logger.info("═" * 60)

        # ── PASO 1: Ingesta ──────────────────────────────────────────
        logger.info("[1/6] Ingesta de datos...")
        df_raw_pd, df_complaints_pd = self._ingest(n_customers)
        df_raw = self.spark.createDataFrame(df_raw_pd)
        df_raw.cache()
        logger.info(f"  RAW: {df_raw.count():,} registros")

        # ── PASO 2: SILVER ───────────────────────────────────────────
        logger.info("[2/6] Transformación SILVER (calidad de datos)...")
        df_silver = self.feature_engineer.raw_to_silver(df_raw)
        df_silver.cache()

        # ── PASO 3: Embeddings BERT ──────────────────────────────────
        if self.run_embeddings and df_complaints_pd is not None:
            logger.info("[3/6] Generando embeddings BERT...")
            df_embeddings = self._run_embeddings(df_complaints_pd)
            df_silver = self._join_embeddings(df_silver, df_embeddings)
        else:
            logger.info("[3/6] Embeddings: SKIP")

        # ── PASO 4: GOLD / Feature Engineering ──────────────────────
        logger.info("[4/6] Feature Engineering GOLD...")
        df_gold = self.feature_engineer.silver_to_gold(df_silver)
        df_ml = self.feature_engineer.fit_transform(df_gold)

        # ── PASO 5: Train / Eval ─────────────────────────────────────
        logger.info("[5/6] Entrenando modelo de Churn...")
        df_train, df_test = df_ml.randomSplit([0.8, 0.2], seed=42)
        self.churn_predictor.train_spark(df_train, cross_validate=False)
        metrics = self.churn_predictor.evaluate_spark(df_test)
        logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f} | Gini: {metrics['gini']:.4f}")

        # ── PASO 6: Scoring masivo ───────────────────────────────────
        logger.info("[6/6] Scoring de todos los clientes...")
        df_scored = self.churn_predictor.score_customers(df_ml)
        df_output = self._add_vip_classification(df_scored)

        # ── Exportar resultados ──────────────────────────────────────
        self._export_results(df_output)

        logger.info("═" * 60)
        logger.info("  PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("═" * 60)
        return df_output

    # ── Métodos privados ─────────────────────────────────────────────────────

    def _ingest(self, n_customers: int):
        """Genera o carga datos del Lakehouse RAW."""
        raw_path = self.data_path / "raw" / "customers.parquet"

        if raw_path.exists():
            logger.info(f"  Cargando datos existentes: {raw_path}")
            df_pd = pd.read_parquet(raw_path)
            complaints_path = self.data_path / "raw" / "complaints.parquet"
            df_complaints = pd.read_parquet(complaints_path) if complaints_path.exists() else None
        else:
            logger.info(f"  Generando datos sintéticos ({n_customers:,} clientes)...")
            gen = BankingDataGenerator(n_customers=n_customers)
            df_pd, df_complaints = gen.save_to_parquet(str(self.data_path / "raw"))

        return df_pd, df_complaints

    def _run_embeddings(self, df_complaints_pd: pd.DataFrame) -> pd.DataFrame:
        """Procesa textos con BERT y retorna embeddings agregados por cliente."""
        customer_emb = self.bert_engine.aggregate_customer_embeddings(
            df_complaints_pd,
            customer_id_col="customer_id",
            text_col="complaint_text",
        )
        # Solo conservar columnas relevantes (evitar curse of dimensionality)
        # En producción usar PCA o reducir a top-N dims
        cols_to_keep = ["customer_id", "churn_intent_score"] + \
                       [c for c in customer_emb.columns if c.startswith("emb_")][:32]
        return customer_emb[cols_to_keep]

    def _join_embeddings(
        self,
        df_silver: DataFrame,
        df_emb: pd.DataFrame,
    ) -> DataFrame:
        """Join del embedding score al DataFrame Spark principal."""
        df_emb_spark = self.spark.createDataFrame(
            df_emb[["customer_id", "churn_intent_score"]]
        )
        return df_silver.join(df_emb_spark, on="customer_id", how="left").fillna(
            {"churn_intent_score": 0.0}
        )

    def _add_vip_classification(self, df: DataFrame) -> DataFrame:
        """Añade clasificación VIP basada en reglas de negocio + ML."""
        return (
            df
            .withColumn(
                "vip_score",
                (
                    F.col("avg_balance_6m") * 0.30 / 1_000_000 +
                    F.col("clv_score") * 0.25 / 1_000_000 +
                    F.col("n_products") * 0.20 / 8.0 +
                    F.col("tenure_months") * 0.15 / 360.0 +
                    F.col("monthly_transactions") * 0.10 / 50.0
                ).cast("double")
            )
            .withColumn(
                "vip_classification",
                F.when(F.col("vip_score") >= 0.7, "PLATINUM")
                .when(F.col("vip_score") >= 0.5, "GOLD")
                .when(F.col("vip_score") >= 0.3, "SILVER")
                .otherwise("STANDARD"),
            )
        )

    def _export_results(self, df: DataFrame) -> None:
        """
        Exporta resultados en formato Parquet/Delta para consumo de Power BI.

        En Databricks: escribir a ADLS Gen2 en formato Delta.
        En local: Parquet en data/gold/.
        """
        output_path = str(self.data_path / "gold" / "churn_vip_scores")
        cols_output = [
            "customer_id", "segment", "region",
            "churn_probability", "churn_risk_level", "recommended_action",
            "vip_score", "vip_classification",
            "avg_balance_6m", "clv_score", "n_products",
            "monthly_transactions", "gold_ts",
        ]

        # Filtrar solo columnas que existen
        available = [c for c in cols_output if c in df.columns]

        (
            df.select(*available)
            .coalesce(1)  # Un archivo en local; en prod usar partitionBy
            .write
            .mode("overwrite")
            .parquet(output_path)
        )
        logger.success(f"Resultados exportados → {output_path}")

        # Resumen ejecutivo
        self._print_summary(df)

    def _print_summary(self, df: DataFrame) -> None:
        """Imprime resumen ejecutivo del pipeline."""
        total = df.count()
        churn_dist = (
            df.groupBy("churn_risk_level")
            .count()
            .orderBy(F.col("count").desc())
            .toPandas()
        )
        vip_dist = (
            df.groupBy("vip_classification")
            .count()
            .orderBy(F.col("count").desc())
            .toPandas()
        )

        logger.info("─── RESUMEN EJECUTIVO ─────────────────────────────────")
        logger.info(f"  Total clientes procesados: {total:,}")
        logger.info("  Distribución de Riesgo Churn:")
        for _, row in churn_dist.iterrows():
            pct = row["count"] / total * 100
            logger.info(f"    {row['churn_risk_level']:<12}: {row['count']:>6,} ({pct:.1f}%)")
        logger.info("  Clasificación VIP:")
        for _, row in vip_dist.iterrows():
            pct = row["count"] / total * 100
            logger.info(f"    {row['vip_classification']:<10}: {row['count']:>6,} ({pct:.1f}%)")
        logger.info("──────────────────────────────────────────────────────")
