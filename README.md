# 🏦 Banco Futura — Sistema de Predicción de Churn y Detección de Clientes VIP

> **Arquitectura enterprise de analítica avanzada sobre Big Data para banca moderna**
>
> **Stack:** PySpark · Azure Databricks · Delta Lake · BERT · FastAPI · Power BI

---

# 📐 Arquitectura General

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        FUENTES DE DATOS                             │
│  Core Bancario  │  CRM  │  App Móvil  │  Canales  │  Call Center    │
└────────────────────────────┬────────────────────────────────────────┘
                             │ CDC / Kafka Events
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   AZURE DATA LAKE STORAGE Gen2                      │
│                                                                     │
│         RAW Zone  │  SILVER Zone  │  GOLD Zone                      │
│         (Ingest)  │  (Cleansed)   │  (Features)                     │
│                                                                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AZURE DATABRICKS + SPARK                         │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐     │
│  │  Feature    │  │  BERT/NLP    │  │   MLflow Experiment     │     │
│  │ Engineering │  │ Embeddings   │  │   Tracking              │     │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘     │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Random Forest + XGBoost + Spark MLlib Ensemble              │   │
│  │ Churn Score │ VIP Score │ SHAP Explainability               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼

       ┌──────────┐  ┌──────────────┐  ┌──────────┐
       │ Power BI │  │   FastAPI    │  │   CRM    │
       │Dashboard │  │ Real-time    │  │ Alertas  │
       │          │  │ Scoring      │  │   VIP    │
       └──────────┘  └──────────────┘  └──────────┘
```

---

# 🏛️ Medallion Architecture (Delta Lake)

| Capa       | Descripción                                                                        | Formato        | Latencia            |
| ---------- | ---------------------------------------------------------------------------------- | -------------- | ------------------- |
| **RAW**    | Datos crudos sin modificar provenientes del core bancario, CRM y canales digitales | Parquet / JSON | Tiempo real o batch |
| **SILVER** | Datos limpios, validados, deduplicados y tipados                                   | Delta          | Micro-batch (5 min) |
| **GOLD**   | Features para Machine Learning, KPIs y métricas agregadas de negocio               | Delta          | Batch diario        |

---

# 📁 Estructura del Repositorio

```text
bank-churn-prediction/
│
├── data/
│   ├── raw/                        # Zona RAW: datos crudos del Lakehouse
│   ├── silver/                     # Zona SILVER: datos limpios y validados
│   └── gold/                       # Zona GOLD: features ML + scores
│
├── notebooks/
│   ├── 01_EDA_customers.ipynb      # Análisis exploratorio clientes
│   ├── 02_feature_analysis.ipynb   # Análisis de features
│   ├── 03_model_evaluation.ipynb   # Evaluación de modelos
│   └── 04_bert_analysis.ipynb      # NLP y embeddings
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── synthetic_data_generator.py
│   │   └── kafka_consumer.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── churn_model.py
│   │   └── vip_model.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── bert_embeddings.py
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   └── churn_pipeline.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── spark_session.py
│
├── api/
│   └── main.py
│
├── dashboards/
│   └── banco_futura_powerbi.pbix
│
├── architecture/
│   └── diagrams/
│
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_churn_model.py
│   └── test_api.py
│
├── config/
│   └── settings.yaml
│
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

# 🚀 Quickstart

## 1. Clonar y configurar entorno

```bash
git clone https://github.com/banco-futura/churn-vip-system.git

cd bank-churn-prediction

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

## 2. Generar datos sintéticos y ejecutar pipeline

```python
from src.pipelines.churn_pipeline import ChurnVIPPipeline

pipeline = ChurnVIPPipeline(
    env="local",
    data_path="data",
    model_type="random_forest",
    run_embeddings=True,
)

df_results = pipeline.run(
    n_customers=10_000
)
```

---

## 3. Levantar API de scoring

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

## 4. Ejemplo de request a la API

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

# 🤖 Modelos de Machine Learning

## Churn Prediction

| Modelo              | Framework              | AUC-ROC Esperado | Uso                     |
| ------------------- | ---------------------- | ---------------- | ----------------------- |
| Random Forest       | Spark MLlib            | 0.82 – 0.87      | Producción batch        |
| XGBoost             | scikit-learn / xgboost | 0.85 – 0.90      | Explicabilidad con SHAP |
| Logistic Regression | Spark MLlib            | 0.75 – 0.80      | Baseline interpretable  |

---

## Features más relevantes (SHAP)

1. `avg_balance_6m` — Saldo promedio últimos 6 meses.
2. `monthly_transactions` — Frecuencia transaccional.
3. `n_complaints_3m` — Reclamos recientes.
4. `nps_score` — Net Promoter Score.
5. `digital_engagement_score` — Interacción digital compuesta.
6. `churn_intent_score` — Score NLP derivado de BERT.
7. `days_since_last_contact` — Recencia de contacto.
8. `clv_score` — Customer Lifetime Value.

---

# 🧠 BERT Embeddings

El módulo NLP transforma texto libre proveniente de reclamos, tickets y comentarios de clientes en señales cuantitativas utilizables por los modelos predictivos.

```python
from src.embeddings.bert_embeddings import BERTEmbeddingEngine

engine = BERTEmbeddingEngine(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

score = engine.compute_churn_intent_score([
    "Quiero cerrar mi cuenta, estoy muy insatisfecho",
    "Excelente servicio, muy contento con el banco"
])

# Resultado esperado:
# [0.87, 0.03]
```

Interpretación:

* **0.87** → Alta intención de abandono.
* **0.03** → Alta probabilidad de retención.

---

# 📊 Power BI Integration

El pipeline exporta resultados hacia la capa GOLD del Lakehouse en formato Delta Lake.

Power BI consume la información mediante **DirectQuery** sobre Azure Data Lake Storage Gen2.

## Tablas disponibles

| Tabla                   | Descripción                                    |
| ----------------------- | ---------------------------------------------- |
| `gold/churn_vip_scores` | Score de churn y clasificación VIP por cliente |
| `gold/risk_summary`     | Resumen de riesgo por segmento y región        |
| `gold/vip_portfolio`    | Cartera de clientes VIP basada en CLV          |

---

# 📏 Roadmap del Proyecto

| Fase       | Descripción                                            | Estado         |
| ---------- | ------------------------------------------------------ | -------------- |
| **Fase 1** | Arquitectura base, datos sintéticos y pipeline PySpark | ✅ Completa     |
| **Fase 2** | Feature Engineering y modelo Random Forest             | ✅ Completa     |
| **Fase 3** | XGBoost, SHAP y MLflow                                 | 🔄 En progreso |
| **Fase 4** | Integración de embeddings BERT                         | 🔄 En progreso |
| **Fase 5** | FastAPI productiva con autenticación JWT               | ⏳ Planificada  |
| **Fase 6** | Dashboard Power BI y alertas CRM                       | ⏳ Planificada  |
| **Fase 7** | CI/CD Azure DevOps y Databricks Jobs                   | ⏳ Planificada  |

---

# 🎯 Objetivos de Negocio

### Reducción de Churn

Identificar clientes con alta probabilidad de abandono antes de que cancelen sus productos financieros.

### Segmentación VIP

Detectar clientes estratégicos mediante CLV, comportamiento transaccional y engagement digital.

### Retención Proactiva

Generar alertas automáticas para campañas de retención personalizadas.

### Explicabilidad

Permitir que las áreas de negocio comprendan por qué un cliente fue clasificado como riesgo o VIP.

---

# 🔐 Consideraciones de Seguridad

* Gestión de secretos mediante Azure Key Vault.
* Encriptación de datos en reposo y en tránsito.
* Control de acceso basado en roles (RBAC).
* Auditoría de pipelines mediante MLflow y Databricks.
* Anonimización de datos sensibles para ambientes de desarrollo.

---

# 🧪 Testing

Ejecutar todos los tests:

```bash
pytest tests/
```

Ejecutar con cobertura:

```bash
pytest --cov=src tests/
```

---

# 📋 Tecnologías

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.5-orange)
![Databricks](https://img.shields.io/badge/Databricks-15.1-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-yellow)
![BERT](https://img.shields.io/badge/BERT-SentenceTransformers-purple)
![MLflow](https://img.shields.io/badge/MLflow-2.13-blue)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-3.1-lightblue)

---

# 👨‍💻 Autor

Proyecto demostrativo de arquitectura moderna de datos y Machine Learning para banca digital utilizando:

* Azure Databricks
* Apache Spark
* Delta Lake
* FastAPI
* Power BI
* MLflow
* BERT / SentenceTransformers

---

# 📄 Licencia

MIT License.
