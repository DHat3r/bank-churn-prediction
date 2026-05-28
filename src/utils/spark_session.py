"""
spark_session.py
────────────────
Factory de SparkSession para Banco Futura.
Soporta ejecución local (dev), Databricks (staging/prod) y tests.

Uso:
    from src.utils.spark_session import SparkSessionFactory
    spark = SparkSessionFactory.get_session(env="local")
"""
from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from loguru import logger
from pyspark.sql import SparkSession


class Environment(str, Enum):
    LOCAL = "local"
    DATABRICKS = "databricks"
    TEST = "test"


class SparkSessionFactory:
    """
    Singleton factory para SparkSession.
    Aplica configuraciones específicas por entorno.
    """

    _instance: Optional[SparkSession] = None

    @classmethod
    def get_session(
        cls,
        env: str = "local",
        app_name: str = "BancoFutura-ChurnVIP",
        force_new: bool = False,
    ) -> SparkSession:
        """
        Retorna (o crea) la SparkSession singleton.

        Args:
            env:        Entorno de ejecución (local | databricks | test).
            app_name:   Nombre de la aplicación Spark.
            force_new:  Si True, destruye la sesión existente y crea una nueva.

        Returns:
            SparkSession configurada para el entorno.
        """
        if cls._instance and not force_new:
            logger.debug("Reutilizando SparkSession existente.")
            return cls._instance

        environment = Environment(env)
        logger.info(f"Creando SparkSession | env={environment.value} | app={app_name}")

        builder = SparkSession.builder.appName(app_name)

        if environment == Environment.LOCAL:
            builder = cls._configure_local(builder)
        elif environment == Environment.DATABRICKS:
            builder = cls._configure_databricks(builder)
        elif environment == Environment.TEST:
            builder = cls._configure_test(builder)

        cls._instance = builder.getOrCreate()
        cls._instance.sparkContext.setLogLevel("WARN")
        logger.success(f"SparkSession iniciada: {cls._instance.version}")
        return cls._instance

    # ── Configuraciones por entorno ──────────────────────────────

    @staticmethod
    def _configure_local(builder: SparkSession.Builder) -> SparkSession.Builder:
        return (
            builder
            .master("local[*]")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .config("spark.sql.shuffle.partitions", "8")          # Ajustar en prod
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.adaptive.enabled", "true")          # AQE activado
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.ui.showConsoleProgress", "false")
        )

    @staticmethod
    def _configure_databricks(builder: SparkSession.Builder) -> SparkSession.Builder:
        """
        En Databricks la sesión ya existe; solo se aplican configs adicionales.
        El cluster tiene Delta Lake y las extensiones preinstaladas.
        """
        return (
            builder
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            .config("spark.databricks.delta.optimizeWrite.enabled", "true")
            .config("spark.databricks.delta.autoCompact.enabled", "true")
        )

    @staticmethod
    def _configure_test(builder: SparkSession.Builder) -> SparkSession.Builder:
        return (
            builder
            .master("local[2]")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
        )

    @classmethod
    def stop(cls) -> None:
        """Detiene la SparkSession activa (usar en teardown de tests)."""
        if cls._instance:
            cls._instance.stop()
            cls._instance = None
            logger.info("SparkSession detenida.")
