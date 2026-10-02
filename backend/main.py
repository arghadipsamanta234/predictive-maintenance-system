
from pathlib import Path
from datetime import datetime, timezone
import sqlite3

import joblib
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "predictive_maintenance_model.joblib"
DB_PATH = BASE_DIR / "data" / "predictive_maintenance.db"

# A reading is classified as a failure when its estimated
# probability reaches this threshold.
FAILURE_THRESHOLD = 0.40

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. Run train_model.py first."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Experimental machine failure prediction API with "
        "prediction history, alerts, and statistics."
    ),
    version="3.0.0",
)


# ============================================================
# 2. DATABASE INITIALIZATION
# ============================================================

def initialize_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                product_type TEXT NOT NULL,
                air_temperature REAL NOT NULL,
                process_temperature REAL NOT NULL,
                rotational_speed REAL NOT NULL,
                torque REAL NOT NULL,
                tool_wear REAL NOT NULL,
                predicted_failure INTEGER NOT NULL,
                failure_probability REAL NOT NULL
            )
        """)


initialize_database()


# ============================================================
# 3. REQUEST VALIDATION
# ============================================================

class MachineReading(BaseModel):
    product_type: str = Field(
        pattern=r"^[LMH]$",
        description="Machine product type: L, M, or H",
    )
    air_temperature: float = Field(ge=250, le=350)
    process_temperature: float = Field(ge=250, le=400)
    rotational_speed: float = Field(gt=0, le=10000)
    torque: float = Field(ge=0, le=1000)
    tool_wear: float = Field(ge=0, le=1000)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def get_failure_probability(input_data: pd.DataFrame) -> float:
    """Return the model probability for the failure class (1)."""

    classifier = model.named_steps.get("classifier")

    if classifier is not None and hasattr(classifier, "classes_"):
        classes = list(classifier.classes_)
    elif hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        raise RuntimeError(
            "Cannot identify the model's failure class."
        )

    if 1 not in classes:
        raise RuntimeError(
            "The trained model does not contain failure class 1."
        )

    failure_index = classes.index(1)
    probabilities = model.predict_proba(input_data)[0]

    return float(probabilities[failure_index])


def get_database_connection():
    """Create a connection to the prediction database."""
    return sqlite3.connect(DB_PATH)


# ============================================================
# 5. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API is running",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "history": "/history",
        "alerts": "/alerts",
        "statistics": "/statistics",
        "failure_threshold": FAILURE_THRESHOLD,
    }


# ============================================================
# 6. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    database_connected = False

    try:
        with get_database_connection() as conn:
            conn.execute("SELECT 1")
        database_connected = True
    except sqlite3.Error:
        database_connected = False

    return {
        "status": "healthy" if database_connected else "degraded",
        "model_loaded": model is not None,
        "database": database_connected,
        "failure_threshold": FAILURE_THRESHOLD,
    }


# ============================================================
# 7. MACHINE FAILURE PREDICTION
# ============================================================

@app.post("/predict")
def predict(reading: MachineReading):
    input_data = pd.DataFrame([{
        "Type": reading.product_type,
        "Air temperature [K]": reading.air_temperature,
        "Process temperature [K]": reading.process_temperature,
        "Rotational speed [rpm]": reading.rotational_speed,
        "Torque [Nm]": reading.torque,
        "Tool wear [min]": reading.tool_wear,
    }])

    # Estimate the probability of failure.
    probability = get_failure_probability(input_data)

    # Apply the configurable classification threshold.
    prediction = int(probability >= FAILURE_THRESHOLD)

    timestamp = datetime.now(timezone.utc).isoformat()

    # Save the prediction and input sensor readings.
    with get_database_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO predictions (
                timestamp,
                product_type,
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear,
                predicted_failure,
                failure_probability
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            reading.product_type,
            reading.air_temperature,
            reading.process_temperature,
            reading.rotational_speed,
            reading.torque,
            reading.tool_wear,
            prediction,
            probability,
        ))

        record_id = cursor.lastrowid

    return {
        "id": record_id,
        "timestamp": timestamp,
        "product_type": reading.product_type,
        "predicted_failure": prediction,
        "estimated_failure_probability": round(probability, 4),
        "estimated_failure_percentage": round(probability * 100, 2),
        "failure_threshold": FAILURE_THRESHOLD,
        "status": (
            "Failure predicted"
            if prediction == 1
            else "No failure predicted"
        ),
        "warning": (
            "Experimental prediction only. Model performance has "
            "not been validated for real industrial safety decisions."
        ),
    }


# ============================================================
# 8. PREDICTION HISTORY
# ============================================================

@app.get("/history")
def history(
    limit: int = Query(default=50, ge=1, le=500),
):
    with get_database_connection() as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT *
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return {
        "count": len(rows),
        "records": [dict(row) for row in rows],
    }


# ============================================================
# 9. FAILURE ALERTS
# ============================================================

@app.get("/alerts")
def alerts(
    limit: int = Query(default=50, ge=1, le=500),
):
    with get_database_connection() as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT *
            FROM predictions
            WHERE predicted_failure = 1
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return {
        "count": len(rows),
        "alerts": [dict(row) for row in rows],
    }


# ============================================================
# 10. OVERALL STATISTICS
# ============================================================

@app.get("/statistics")
def statistics():
    with get_database_connection() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) AS total_predictions,

                COALESCE(SUM(
                    CASE
                        WHEN predicted_failure = 0 THEN 1
                        ELSE 0
                    END
                ), 0) AS normal_predictions,

                COALESCE(SUM(
                    CASE
                        WHEN predicted_failure = 1 THEN 1
                        ELSE 0
                    END
                ), 0) AS failure_predictions,

                COALESCE(
                    AVG(failure_probability), 0
                ) AS average_risk

            FROM predictions
        """).fetchone()

    return {
        "total_predictions": row[0],
        "normal_predictions": row[1],
        "failure_predictions": row[2],
        "average_estimated_risk_percentage": round(
            row[3] * 100, 2
        ),
        "failure_threshold": FAILURE_THRESHOLD,
    }
