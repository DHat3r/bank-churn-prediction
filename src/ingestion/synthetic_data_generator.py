"""
synthetic_data_generator.py
────────────────────────────
Genera dataset sintético realista de clientes bancarios para desarrollo/demo.

Simula las tablas que llegarían desde:
  - Core bancario (transacciones, saldos)
  - CRM (perfil cliente, segmentos)
  - Canales digitales (app, web)
  - Centro de atención (tickets, reclamos)

El dataset incluye señales reales de churn y características VIP.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ── Semillas para reproducibilidad ──────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class BankingDataGenerator:
    """
    Genera datos sintéticos de clientes bancarios.

    Args:
        n_customers:  Número de clientes a generar.
        churn_rate:   Tasa de churn objetivo (0.0 – 1.0).
        vip_rate:     Proporción de clientes VIP (0.0 – 1.0).
    """

    REGIONS = ["RM", "Valparaíso", "Biobío", "Maule", "Los Lagos", "Araucanía"]
    SEGMENTS = ["Retail", "PyME", "Premium", "Universitario", "Jubilado"]
    PRODUCTS = ["cuenta_vista", "cuenta_corriente", "credito_hipotecario",
                "tarjeta_credito", "credito_consumo", "deposito_plazo",
                "fondos_mutuos", "seguro_vida"]
    COMPLAINT_TEMPLATES = [
        "Tuve un problema con mi transferencia y nadie me ayudó.",
        "La aplicación no funciona correctamente hace semanas.",
        "Quiero cerrar mi cuenta, el servicio es terrible.",
        "Me cobraron comisiones que no corresponden.",
        "Excelente servicio, muy satisfecho con el banco.",
        "El ejecutivo fue muy amable y resolvió mi problema.",
        "No puedo acceder a mi cuenta desde la app.",
        "Necesito información sobre mis créditos disponibles.",
        "Estoy evaluando cambiarme a otro banco.",
        "El proceso de préstamo fue muy rápido, gracias.",
    ]

    def __init__(
        self,
        n_customers: int = 50_000,
        churn_rate: float = 0.18,
        vip_rate: float = 0.08,
    ) -> None:
        self.n = n_customers
        self.churn_rate = churn_rate
        self.vip_rate = vip_rate
        logger.info(
            f"Inicializando generador | clientes={n_customers:,} "
            f"| churn={churn_rate:.0%} | VIP={vip_rate:.0%}"
        )

    # ── API pública ──────────────────────────────────────────────────────────

    def generate_customers(self) -> pd.DataFrame:
        """Tabla principal de clientes (equivalente a capa RAW del Lakehouse)."""
        logger.info("Generando tabla de clientes...")

        ids = [f"CLI-{str(i).zfill(7)}" for i in range(1, self.n + 1)]
        is_vip = np.random.random(self.n) < self.vip_rate

        # Los VIP tienen menor probabilidad de churn
        churn_prob = np.where(
            is_vip,
            self.churn_rate * 0.3,
            self.churn_rate * 1.2,
        )
        is_churn = np.random.random(self.n) < churn_prob

        # Características demográficas
        ages = np.random.normal(42, 14, self.n).clip(18, 80).astype(int)
        tenure_months = np.random.exponential(48, self.n).clip(1, 360).astype(int)
        regions = np.random.choice(self.REGIONS, self.n)
        segments = np.random.choice(
            self.SEGMENTS, self.n,
            p=[0.45, 0.20, 0.15, 0.12, 0.08],
        )

        # Saldo y métricas financieras (VIP tienen valores más altos)
        base_balance = np.where(
            is_vip,
            np.random.lognormal(13, 1.5, self.n),    # ~$440K avg
            np.random.lognormal(10, 1.2, self.n),    # ~$22K avg
        )
        # Churn → saldo en declive
        balance_trend = np.where(is_churn, -0.3, 0.05)
        avg_balance_6m = (base_balance * (1 + balance_trend)).clip(0)

        # Métricas transaccionales
        monthly_txn = np.where(
            is_churn,
            np.random.poisson(3, self.n),
            np.random.poisson(18, self.n),
        ).clip(0)

        app_logins_30d = np.where(
            is_churn,
            np.random.poisson(2, self.n),
            np.random.poisson(12, self.n),
        ).clip(0)

        n_products = np.where(
            is_vip,
            np.random.randint(4, 9, self.n),
            np.random.randint(1, 5, self.n),
        )

        # Customer Lifetime Value (simplificado)
        clv = (avg_balance_6m * 0.015 + monthly_txn * 500 + n_products * 2000).clip(0)

        # Reclamos (señal fuerte de churn)
        n_complaints_3m = np.where(
            is_churn,
            np.random.poisson(2.5, self.n),
            np.random.poisson(0.3, self.n),
        ).clip(0)

        # NPS Score (señal churn)
        nps_score = np.where(
            is_churn,
            np.random.randint(0, 5, self.n),
            np.random.randint(6, 11, self.n),
        )

        # Fecha último contacto
        today = datetime.now()
        days_since_contact = np.where(
            is_churn,
            np.random.randint(60, 180, self.n),
            np.random.randint(1, 30, self.n),
        )
        last_contact = [
            (today - timedelta(days=int(d))).strftime("%Y-%m-%d")
            for d in days_since_contact
        ]

        df = pd.DataFrame({
            "customer_id": ids,
            "age": ages,
            "tenure_months": tenure_months,
            "region": regions,
            "segment": segments,
            "avg_balance_6m": avg_balance_6m.round(2),
            "monthly_transactions": monthly_txn,
            "app_logins_30d": app_logins_30d,
            "n_products": n_products,
            "clv_score": clv.round(2),
            "n_complaints_3m": n_complaints_3m,
            "nps_score": nps_score,
            "days_since_last_contact": days_since_contact,
            "last_contact_date": last_contact,
            "is_vip": is_vip.astype(int),
            "churn_label": is_churn.astype(int),
            "ingestion_ts": datetime.now().isoformat(),
        })

        logger.success(
            f"Dataset generado: {len(df):,} clientes | "
            f"churn real={df['churn_label'].mean():.1%} | "
            f"VIP real={df['is_vip'].mean():.1%}"
        )
        return df

    def generate_complaints_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera texto de reclamos/comentarios por cliente.
        Estos textos serán procesados por BERT en la capa de embeddings.
        """
        logger.info("Generando textos de reclamos para NLP...")

        records = []
        for _, row in df[df["n_complaints_3m"] > 0].iterrows():
            n = int(row["n_complaints_3m"])
            for _ in range(n):
                # Clientes churn → templates con intención de abandono
                if row["churn_label"] == 1:
                    pool = [t for t in self.COMPLAINT_TEMPLATES if any(
                        kw in t.lower()
                        for kw in ["cerrar", "cambiarme", "terrible", "problema", "no funciona"]
                    )]
                    pool = pool or self.COMPLAINT_TEMPLATES[:5]
                else:
                    pool = self.COMPLAINT_TEMPLATES

                records.append({
                    "customer_id": row["customer_id"],
                    "complaint_text": random.choice(pool),
                    "complaint_date": (
                        datetime.now() - timedelta(days=random.randint(1, 90))
                    ).strftime("%Y-%m-%d"),
                    "channel": random.choice(["app", "call_center", "email", "sucursal"]),
                    "churn_label": row["churn_label"],
                })

        result = pd.DataFrame(records)
        logger.success(f"Textos generados: {len(result):,} registros de reclamos")
        return result

    def save_to_parquet(self, output_dir: str = "data/raw") -> None:
        """Guarda datasets en formato Parquet (capa RAW del Lakehouse)."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        customers = self.generate_customers()
        customers.to_parquet(f"{output_dir}/customers.parquet", index=False)
        logger.success(f"Guardado: {output_dir}/customers.parquet")

        complaints = self.generate_complaints_text(customers)
        complaints.to_parquet(f"{output_dir}/complaints.parquet", index=False)
        logger.success(f"Guardado: {output_dir}/complaints.parquet")

        return customers, complaints


if __name__ == "__main__":
    gen = BankingDataGenerator(n_customers=10_000, churn_rate=0.18, vip_rate=0.08)
    gen.save_to_parquet("data/raw")
