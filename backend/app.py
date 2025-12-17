import json
import os
import time
from datetime import datetime, timezone

import joblib
from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy import create_engine, text
from starlette.responses import JSONResponse, PlainTextResponse, Response

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://feature_user:feature_pass@localhost:5432/feature_store",
)

REQ_COUNT = Counter("forward_requests_total", "Total /forward requests", ["status"])
REQ_LAT = Histogram("forward_request_seconds", "Latency of /forward requests")


app = FastAPI(title="Airline Tweet Sentiment")


def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def init_db(engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        text TEXT NOT NULL,
        label TEXT NOT NULL,
        proba_json JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def load_model(path: str):
    return joblib.load(path)


engine = get_engine()
init_db(engine)

try:
    model = load_model(MODEL_PATH)
except Exception:
    model = None


@app.get("/health")
def health():
    if model is None:
        return JSONResponse({"status": "no_model"}, status_code=503)
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/forward")
async def forward(request: Request):
    start = time.time()

    try:
        payload = await request.json()
    except Exception:
        REQ_COUNT.labels(status="400").inc()
        return PlainTextResponse("bad request", status_code=400)

    if not isinstance(payload, dict) or "text" not in payload or not isinstance(payload["text"], str):
        REQ_COUNT.labels(status="400").inc()
        return PlainTextResponse("bad request", status_code=400)

    text_in = payload["text"].strip()
    if not text_in:
        REQ_COUNT.labels(status="400").inc()
        return PlainTextResponse("bad request", status_code=400)

    if model is None:
        REQ_COUNT.labels(status="403").inc()
        return PlainTextResponse("the model could not process the data", status_code=403)

    try:
        proba = model.predict_proba([text_in])[0].tolist()
        classes = list(getattr(model, "classes_", []))
        if not classes:
            classes = list(model.named_steps["clf"].classes_)
        label = classes[int(max(range(len(proba)), key=lambda i: proba[i]))]
        proba_map = {classes[i]: float(proba[i]) for i in range(len(classes))}
    except Exception:
        REQ_COUNT.labels(status="403").inc()
        return PlainTextResponse("the model could not process the data", status_code=403)

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO predictions(text, label, proba_json, created_at)
                    VALUES (:text, :label, CAST(:proba_json AS JSONB), :created_at);
                    """
                ),
                {
                    "text": text_in,
                    "label": label,
                    "proba_json": json.dumps(proba_map, ensure_ascii=False),
                    "created_at": datetime.now(timezone.utc),
                },
            )
    except Exception:
        pass

    elapsed = time.time() - start
    REQ_LAT.observe(elapsed)
    REQ_COUNT.labels(status="200").inc()

    return JSONResponse({"label": label, "proba": proba_map})
