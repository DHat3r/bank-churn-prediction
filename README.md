# 🏦 Banco Futura — Sistema de Predicción de Churn y Detección de Clientes VIP

> **Arquitectura enterprise de analítica avanzada sobre Big Data para banca moderna**
> Stack: PySpark · Azure Databricks · Delta Lake · BERT · FastAPI · Power BI

---

## 📐 Arquitectura General

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FUENTES DE DATOS                             │
│  Core Bancario  │  CRM  │  App Móvil  │  Canales  │  Call Center    │
└────────────────────────────┬────────────────────────────────────────┘
                             │ CDC / Kafka Events
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   AZURE DATA LAKE STORAGE Gen2                      │
│         RAW Zone  │  SILVER Zone  │  GOLD Zone                      │
│         (Ingest)  │  (Cleansed)   │  (Features)                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AZURE DATABRICKS + SPARK                         │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐     │
│  │  Feature    │  │  BERT/NLP    │  │   MLflow Experiment     │     │
│  │  Engineering│  │  Embeddings  │  │   Tracking              │     │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘     │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │        Random Forest + XGBoost + Spark MLlib Ensemble        │   │
│  │        Churn Score │ VIP Score │ SHAP Explainability         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       ┌──────────┐  ┌──────────────┐  ┌──────────┐
       │ Power BI │  │  FastAPI     │  │   CRM    │
       │ Dashboard│  │  Real-time   │  │  Alertas │
       │          │  │  Scoring     │  │  VIP     │
       └──────────┘  └──────────────┘  └──────────┘
```

---

## 🏛️ Medallion Architecture (Delta Lake)

| Capa | Descripción | Formato | Latencia |
|------|-------------|---------|----------|
| **RAW** | Datos crudos sin modificar del core bancario, CRM, canales | Parquet/JSON | Tiempo real / batch |
| **SILVER** | Datos limpios, schema validado, deduplicados, tipados | Delta | Micro-batch (5 min) |
| **GOLD** | Features de ML, métricas de negocio, KPIs agregados | Delta | Batch diario |

---

## 📁 Estructura del Repositorio

```
bank-churn-prediction/
│
├── data/
│   ├── raw/                        # Zona RAW: datos crudos del Lakehouse
│   ├── silver/                     # Zona SILVER: datos limpios y validados
│   └── gold/                       # Zona GOLD: features ML + scores
│
├── notebooks/
│   ├── 01_EDA_customers.ipynb      # Análisis exploratorio clientes
│   ├── 02_feature_analysis.ipynb   # Análisis de features e importancia
│   ├── 03_model_evaluation.ipynb   # Evaluación y comparación de modelos
│   └── 04_bert_analysis.ipynb      # Análisis NLP y embeddings
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── synthetic_data_generator.py   # Generador datos sintéticos
│   │   └── kafka_consumer.py             # Consumidor Kafka (eventos CDC)
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py        # Pipeline RAW→SILVER→GOLD
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── churn_model.py                # Predictor Churn (RF + XGBoost)
│   │   └── vip_model.py                  # Clasificador VIP (CLV-based)
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── bert_embeddings.py            # BERT/SentenceTransformers engine
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   └── churn_pipeline.py             # Orquestador pipeline completo
│   │
│   └── utils/
│       ├── __init__.py
│       └── spark_session.py              # Factory SparkSession
│
├── api/
│   └── main.py                           # FastAPI: scoring real-time
│
├── dashboards/
│   └── banco_futura_powerbi.pbix         # Dashboard Power BI
│
├── architecture/
│   └── diagrams/                         # Diagramas de arquitectura
│
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_churn_model.py
│   └── test_api.py
│
├── config/
│   └── settings.yaml                     # Configuración por entorno
│
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quickstart

### 1. Clonar y configurar entorno

```bash
git clone https://github.com/banco-futura/churn-vip-system.git
cd bank-churn-prediction

python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Generar datos sintéticos y ejecutar pipeline

```python
from src.pipelines.churn_pipeline import ChurnVIPPipeline

pipeline = ChurnVIPPipeline(
    env="local",
    data_path="data",
    model_type="random_forest",
    run_embeddings=True,
)
df_results = pipeline.run(n_customers=10_000)
```

### 3. Levantar API de scoring

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

### 4. Ejemplo de request a la API

```bash
curl -X POST "http://localhost:8000/predict/churn" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CLI-0001234",
    "age": 42,
    "tenure_months": 36,
    "region": "RM",
    "segment": "Premium",
    "avg_balance_6m": 5500000,
    "monthly_transactions": 3,
    "app_logins_30d": 1,
    "n_products": 2,
    "clv_score": 320000,
    "n_complaints_3m": 2,
    "nps_score": 4,
    "days_since_last_contact": 75,
    "last_complaint_text": "Quiero cerrar mi cuenta, el servicio es muy malo"
  }'
```

---

## 🤖 Modelos de Machine Learning

### Churn Prediction

| Modelo | Framework | AUC-ROC esperado | Uso |
|--------|-----------|-----------------|-----|
| Random Forest | Spark MLlib | 0.82–0.87 | Producción batch |
| XGBoost | scikit-learn / xgb | 0.85–0.90 | SHAP + explicabilidad |
| Logistic Regression | Spark MLlib | 0.75–0.80 | Baseline / interpretable |

### Features más relevantes (por importancia SHAP)

1. `avg_balance_6m` — Saldo promedio últimos 6 meses
2. `monthly_transactions` — Frecuencia transaccional
3. `n_complaints_3m` — Reclamos recientes
4. `nps_score` — Net Promoter Score
5. `digital_engagement_score` — Interacción digital compuesta
6. `churn_intent_score` — Score BERT de intención de abandono (NLP)
7. `days_since_last_contact` — Recencia de contacto
8. `clv_score` — Customer Lifetime Value

---

## 🧠 BERT Embeddings

El módulo NLP transforma texto de reclamos en señales cuantitativas:

```python
from src.embeddings.bert_embeddings import BERTEmbeddingEngine

engine = BERTEmbeddingEngine(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

score = engine.compute_churn_intent_score([
    "Quiero cerrar mi cuenta, estoy muy insatisfecho",
    "Excelente servicio, muy contento con el banco",
])
# → [0.87, 0.03]  — Alta intención churn vs. retención
```

---

## 📊 Power BI Integration

El pipeline exporta los resultados a la capa GOLD del Lakehouse (Parquet/Delta).
Power BI se conecta vía **DirectQuery** al Azure Data Lake:

**Tablas disponibles para Power BI:**
- `gold/churn_vip_scores` — Score por cliente con segmentación
- `gold/risk_summary` — Resumen de riesgo por segmento/región
- `gold/vip_portfolio` — Cartera de clientes VIP con CLV

---

## 📏 Roadmap del Proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| **Fase 1** | Arquitectura base, datos sintéticos, pipeline PySpark | ✅ Completa    |
| **Fase 2** | Feature Engineering avanzado, modelo Random Forest    | ✅ Completa    |
| **Fase 3** | XGBoost + SHAP explainability, MLflow tracking        | 🔄 En progreso |
| **Fase 4** | BERT embeddings integrados al pipeline churn          | 🔄 En progreso |
| **Fase 5** | FastAPI production-ready + autenticación JWT          | ⏳ Planificada |
| **Fase 6** | Dashboard Power BI completo + alertas CRM             | ⏳ Planificada |
| **Fase 7** | Deploy Databricks Jobs + CI/CD Azure DevOps           | ⏳ Planificada |

---

## 📋 Tecnologías

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.5-orange)
![Databricks](https://img.shields.io/badge/Databricks-15.1-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-yellow)
![BERT](https://img.shields.io/badge/BERT-SentenceTransformers-purple)
![MLflow](https://img.shields.io/badge/MLflow-2.13-blue)
![Delta](https://img.shields.io/badge/Delta_Lake-3.1-lightblue)
