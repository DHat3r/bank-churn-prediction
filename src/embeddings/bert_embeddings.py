"""
bert_embeddings.py
──────────────────
Módulo de embeddings BERT para análisis de texto bancario en Banco Futura.

Propósito:
  Transformar comentarios, reclamos y tickets de clientes en vectores
  semánticos que enriquecen el modelo de churn con señales de intención.

Modelos soportados:
  - paraphrase-multilingual-MiniLM-L12-v2  (multilingüe, eficiente, recomendado)
  - dccuchile/bert-base-spanish-wwm-cased   (BETO, español específico)
  - sentence-transformers/all-mpnet-base-v2 (inglés, máxima calidad)

Capacidades:
  - Generación de embeddings en batch con GPU/CPU
  - Distribución con Spark UDFs (Pandas UDF)
  - Detección de intención de abandono por similitud coseno
  - Clustering de clientes por comportamiento textual
  - Export de embeddings como features adicionales al modelo churn
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import torch
    from sentence_transformers import SentenceTransformer, util
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch/sentence-transformers no disponible. Modo mock activado.")


# ── Templates de intención de abandono ──────────────────────────────────────
CHURN_INTENT_ANCHORS = [
    "quiero cerrar mi cuenta",
    "me voy a cambiar de banco",
    "estoy muy insatisfecho con el servicio",
    "voy a cancelar todos mis productos",
    "el banco me ha decepcionado completamente",
    "buscaré mejores condiciones en otro banco",
    "no recomendaría este banco a nadie",
    "quiero hablar con alguien para retirar mi dinero",
]

RETENTION_INTENT_ANCHORS = [
    "muy satisfecho con el servicio",
    "excelente atención al cliente",
    "recomendaría el banco a mis conocidos",
    "el banco resolvió mi problema rápidamente",
    "estoy muy contento con mis productos",
]


class BERTEmbeddingEngine:
    """
    Motor de embeddings BERT para señales textuales de churn.

    Args:
        model_name:  Modelo HuggingFace / SentenceTransformers a cargar.
        device:      "cuda" | "cpu" | "auto"
        batch_size:  Tamaño de batch para inference.
    """

    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Optional[SentenceTransformer] = None

        if device == "auto":
            self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        logger.info(f"BERTEmbeddingEngine | modelo={model_name} | device={self.device}")

    def load_model(self) -> None:
        """Carga el modelo en memoria (lazy loading)."""
        if self._model is not None:
            return
        if not TORCH_AVAILABLE:
            logger.warning("Modo mock: retornando embeddings aleatorios.")
            return

        logger.info(f"Cargando modelo {self.model_name}...")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.success(
            f"Modelo cargado | embedding_dim={self._model.get_sentence_embedding_dimension()}"
        )

    # ── Generación de embeddings ─────────────────────────────────────────────

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings para una lista de textos.

        Args:
            texts: Lista de strings a vectorizar.

        Returns:
            np.ndarray de shape (len(texts), embedding_dim)
        """
        self.load_model()

        if not TORCH_AVAILABLE or self._model is None:
            # Modo mock para desarrollo sin GPU
            dim = 384
            return np.random.randn(len(texts), dim).astype(np.float32)

        logger.info(f"Generando embeddings para {len(texts):,} textos...")
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # Normalizar facilita similitud coseno
            convert_to_numpy=True,
        )
        logger.success(f"Embeddings generados: {embeddings.shape}")
        return embeddings

    # ── Análisis de intención de abandono ────────────────────────────────────

    def compute_churn_intent_score(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Calcula un score de intención de abandono por similitud semántica.

        Para cada texto:
          1. Genera embedding del texto
          2. Calcula similitud coseno con cada anchor de churn
          3. Retorna el score máximo (0-1)

        Args:
            texts: Lista de comentarios/reclamos.

        Returns:
            np.ndarray de scores entre 0 y 1.
        """
        self.load_model()

        text_embeddings = self.encode(texts)
        churn_embeddings = self.encode(CHURN_INTENT_ANCHORS)
        retention_embeddings = self.encode(RETENTION_INTENT_ANCHORS)

        # Similitud coseno: (n_texts, n_anchors)
        if TORCH_AVAILABLE and self._model is not None:
            churn_sim = util.cos_sim(text_embeddings, churn_embeddings).numpy()
            retention_sim = util.cos_sim(text_embeddings, retention_embeddings).numpy()
        else:
            # Mock: similitud aleatoria
            n = len(texts)
            churn_sim = np.random.rand(n, len(CHURN_INTENT_ANCHORS))
            retention_sim = np.random.rand(n, len(RETENTION_INTENT_ANCHORS))

        churn_score = churn_sim.max(axis=1)       # Max similitud con cualquier anchor churn
        retention_score = retention_sim.max(axis=1)

        # Score neto: churn - retención (clampado a [0, 1])
        net_churn_intent = np.clip(churn_score - retention_score * 0.5, 0, 1)
        return net_churn_intent

    # ── Integración con Spark (Pandas UDF) ──────────────────────────────────

    def get_spark_udf(self, spark):
        """
        Retorna un Pandas UDF que genera el churn_intent_score en Spark.

        Uso:
            udf_fn = engine.get_spark_udf(spark)
            df = df.withColumn("churn_intent_score", udf_fn(F.col("complaint_text")))
        """
        from pyspark.sql.functions import pandas_udf
        from pyspark.sql.types import FloatType

        engine = self  # Capturar en closure

        @pandas_udf(FloatType())
        def churn_intent_udf(texts: pd.Series) -> pd.Series:
            scores = engine.compute_churn_intent_score(texts.tolist())
            return pd.Series(scores.astype(np.float32))

        return churn_intent_udf

    # ── Agregación de embeddings por cliente ─────────────────────────────────

    def aggregate_customer_embeddings(
        self,
        df_complaints: pd.DataFrame,
        customer_id_col: str = "customer_id",
        text_col: str = "complaint_text",
    ) -> pd.DataFrame:
        """
        Agrega múltiples textos de un cliente en un único embedding
        usando promedio ponderado.

        Returns:
            DataFrame con customer_id + embedding_dim columnas numéricas
            + churn_intent_score promedio por cliente.
        """
        logger.info("Agregando embeddings por cliente...")

        all_texts = df_complaints[text_col].tolist()
        all_embeddings = self.encode(all_texts)
        intent_scores = self.compute_churn_intent_score(all_texts)

        df_complaints = df_complaints.copy()
        df_complaints["churn_intent_score"] = intent_scores

        # Columnas de embedding
        dim = all_embeddings.shape[1]
        emb_df = pd.DataFrame(
            all_embeddings,
            columns=[f"emb_{i}" for i in range(dim)],
        )
        df_with_emb = pd.concat([df_complaints.reset_index(drop=True), emb_df], axis=1)

        # Agrupar por cliente: promedio de embeddings + max intent score
        emb_cols = [f"emb_{i}" for i in range(dim)]
        agg_dict = {col: "mean" for col in emb_cols}
        agg_dict["churn_intent_score"] = "max"

        customer_embeddings = (
            df_with_emb
            .groupby(customer_id_col)
            .agg(agg_dict)
            .reset_index()
        )

        logger.success(
            f"Embeddings agregados para {len(customer_embeddings):,} clientes | "
            f"dim={dim}"
        )
        return customer_embeddings

    def get_embedding_dimension(self) -> int:
        """Retorna la dimensión del vector de embedding del modelo."""
        self.load_model()
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return 384  # Default MiniLM
