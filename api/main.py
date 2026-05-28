"""
api/main.py
───────────
API REST de inferencia en tiempo real para Banco Futura.

Endpoints:
  POST /predict/churn          → Score de churn para un cliente
  POST /predict/churn/batch    → Score masivo (hasta 1000 clientes)
  GET  /customers/{id}/profile → Perfil completo con score y recomendaciones
  GET  /health                 → Health check
  GET  /metrics                → Métricas del modelo en producción

Diseñada para integrarse con:
  - CRM del banco (webhooks)
  - Core bancario (eventos CDC)
  - Dashboard Power BI (vía Direct Query)
"""
from __future__ import annotations

import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from loguru import logger
import uvicorn


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Banco Futura – Churn & VIP API",
    description=(
        "API de predicción de churn y clasificación VIP para Banco Futura. "
        "Powered by Spark MLlib + BERT Embeddings."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas de Request/Response ──────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """Features de entrada para predicción de churn."""

    customer_id: str = Field(..., description="ID único del cliente")
    age: int = Field(..., ge=18, le=100)
    tenure_months: int = Field(..., ge=0)
    region: str = Field(..., description="Región del cliente")
    segment: str = Field(..., description="Segmento bancario")
    avg_balance_6m: float = Field(..., ge=0, description="Saldo promedio 6 meses")
    monthly_transactions: int = Field(..., ge=0)
    app_logins_30d: int = Field(..., ge=0)
    n_products: int = Field(..., ge=1, le=20)
    clv_score: float = Field(..., ge=0)
    n_complaints_3m: int = Field(..., ge=0)
    nps_score: int = Field(..., ge=0, le=10)
    days_since_last_contact: int = Field(..., ge=0)
    last_complaint_text: Optional[str] = Field(
        None,
        description="Último texto de reclamo para análisis BERT"
    )

    @field_validator("segment")
    @classmethod
    def validate_segment(cls, v: str) -> str:
        valid = {"Retail", "PyME", "Premium", "Universitario", "Jubilado"}
        if v not in valid:
            raise ValueError(f"Segmento inválido. Válidos: {valid}")
        return v


class ChurnPrediction(BaseModel):
    """Respuesta de predicción de churn."""

    customer_id: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_risk_level: str        # ALTO | MEDIO | BAJO | SIN_RIESGO
    recommended_action: str
    vip_classification: str      # PLATINUM | GOLD | SILVER | STANDARD
    vip_score: float
    churn_intent_from_text: Optional[float]
    top_risk_factors: List[str]
    inference_time_ms: float
    model_version: str


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., max_length=1000)


class BatchResponse(BaseModel):
    total: int
    predictions: List[ChurnPrediction]
    batch_inference_time_ms: float


# ── Lógica de negocio (mock de modelo – en prod se conecta a MLflow registry) ──

MODEL_VERSION = "v1.2.3-spark-rf"

CHURN_THRESHOLDS = {"alto": 0.70, "medio": 0.40, "bajo": 0.20}

def _compute_churn_score(features: CustomerFeatures) -> dict:
    """
    Simula inferencia del modelo en producción.
    En real: cargar modelo desde MLflow registry y llamar .transform()
    """
    import random, math

    # Señales de riesgo heurísticas (en prod → modelo ML)
    risk_signals = []
    score = 0.0

    if features.avg_balance_6m < 100_000:
        score += 0.15
        risk_signals.append("Saldo bajo")

    if features.monthly_transactions < 5:
        score += 0.20
        risk_signals.append("Baja actividad transaccional")

    if features.app_logins_30d < 3:
        score += 0.15
        risk_signals.append("Baja interacción digital")

    if features.n_complaints_3m >= 2:
        score += 0.25
        risk_signals.append("Múltiples reclamos recientes")

    if features.nps_score <= 4:
        score += 0.20
        risk_signals.append("NPS crítico (Detractor)")

    if features.days_since_last_contact > 60:
        score += 0.10
        risk_signals.append("Sin contacto reciente")

    if features.n_products == 1:
        score += 0.10
        risk_signals.append("Producto único (baja vinculación)")

    score = min(score + random.uniform(-0.05, 0.05), 1.0)
    score = max(score, 0.0)

    # VIP score
    vip = (
        min(features.avg_balance_6m / 10_000_000, 1.0) * 0.35 +
        min(features.clv_score / 2_000_000, 1.0) * 0.30 +
        min(features.n_products / 8, 1.0) * 0.20 +
        min(features.tenure_months / 240, 1.0) * 0.15
    )

    return {
        "churn_score": round(score, 4),
        "vip_score": round(vip, 4),
        "risk_signals": risk_signals[:3],
    }


def _classify_risk(score: float) -> tuple[str, str]:
    if score >= CHURN_THRESHOLDS["alto"]:
        return "ALTO", "LLAMADA_EJECUTIVO"
    elif score >= CHURN_THRESHOLDS["medio"]:
        return "MEDIO", "CAMPAÑA_RETENCIÓN"
    elif score >= CHURN_THRESHOLDS["bajo"]:
        return "BAJO", "OFERTA_DIGITAL"
    return "SIN_RIESGO", "MONITOREO"


def _classify_vip(vip_score: float) -> str:
    if vip_score >= 0.7: return "PLATINUM"
    if vip_score >= 0.5: return "GOLD"
    if vip_score >= 0.3: return "SILVER"
    return "STANDARD"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Infra"])
async def health_check():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "timestamp": time.time(),
    }


@app.post("/predict/churn", response_model=ChurnPrediction, tags=["Predicción"])
async def predict_churn(customer: CustomerFeatures):
    """
    Predice el riesgo de churn para un cliente individual.

    - **churn_probability**: Score de 0 a 1 (mayor = mayor riesgo)
    - **churn_risk_level**: Clasificación de riesgo operacional
    - **recommended_action**: Acción sugerida para el equipo comercial
    - **top_risk_factors**: Principales factores de riesgo detectados
    """
    start = time.perf_counter()

    result = _compute_churn_score(customer)
    churn_score = result["churn_score"]
    risk_level, action = _classify_risk(churn_score)
    vip_class = _classify_vip(result["vip_score"])

    # Simular BERT intent score si hay texto
    bert_score = None
    if customer.last_complaint_text:
        # En producción: llamar a BERTEmbeddingEngine.compute_churn_intent_score()
        bert_score = round(min(churn_score * 1.1 + 0.05, 1.0), 4)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return ChurnPrediction(
        customer_id=customer.customer_id,
        churn_probability=churn_score,
        churn_risk_level=risk_level,
        recommended_action=action,
        vip_classification=vip_class,
        vip_score=result["vip_score"],
        churn_intent_from_text=bert_score,
        top_risk_factors=result["risk_signals"],
        inference_time_ms=round(elapsed_ms, 2),
        model_version=MODEL_VERSION,
    )


@app.post("/predict/churn/batch", response_model=BatchResponse, tags=["Predicción"])
async def predict_churn_batch(request: BatchRequest):
    """
    Scoring masivo de hasta 1.000 clientes por request.
    Ideal para campañas nocturnas del CRM.
    """
    start = time.perf_counter()
    predictions = []

    for customer in request.customers:
        result = _compute_churn_score(customer)
        churn_score = result["churn_score"]
        risk_level, action = _classify_risk(churn_score)
        vip_class = _classify_vip(result["vip_score"])

        predictions.append(ChurnPrediction(
            customer_id=customer.customer_id,
            churn_probability=churn_score,
            churn_risk_level=risk_level,
            recommended_action=action,
            vip_classification=vip_class,
            vip_score=result["vip_score"],
            churn_intent_from_text=None,
            top_risk_factors=result["risk_signals"],
            inference_time_ms=0,
            model_version=MODEL_VERSION,
        ))

    elapsed_ms = (time.perf_counter() - start) * 1000
    return BatchResponse(
        total=len(predictions),
        predictions=predictions,
        batch_inference_time_ms=round(elapsed_ms, 2),
    )


@app.get("/customers/{customer_id}/profile", tags=["Clientes"])
async def get_customer_profile(customer_id: str):
    """
    Retorna perfil completo de un cliente con scoring y recomendaciones.
    En producción consulta Delta Lake (Gold layer).
    """
    # Placeholder: en prod → consultar Delta Lake / cache Redis
    raise HTTPException(
        status_code=501,
        detail=f"Endpoint en desarrollo. Conectar a Delta Lake Gold layer para {customer_id}.",
    )


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
