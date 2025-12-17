import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy import create_engine, text
from starlette.responses import JSONResponse, PlainTextResponse, Response

from ml import normalize_text

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.onnx")
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


def load_session(path: str):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


engine = get_engine()
init_db(engine)

try:
    session = load_session(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    meta = session.get_modelmeta().custom_metadata_map or {}
except Exception:
    session = None
    input_name = None
    meta = {}


@app.get("/health")
def health():
    if session is None:
        return JSONResponse({"status": "no_model"}, status_code=503)
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metadata")
def metadata():
    return {
        "commit_hash": meta.get("commit_hash", "unknown"),
        "saved_at": meta.get("saved_at", "unknown"),
        "experiment_name": meta.get("experiment_name", "unknown"),
    }


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
    text_in = normalize_text(text_in)

    if not text_in:
        REQ_COUNT.labels(status="400").inc()
        return PlainTextResponse("bad request", status_code=400)

    if session is None or input_name is None:
        REQ_COUNT.labels(status="403").inc()
        return PlainTextResponse("the model could not process the data", status_code=403)

    try:
        x = np.array([[text_in]], dtype=object)
        outputs = session.run(None, {input_name: x})

        proba_vec = outputs[1][0].tolist()

        meta = session.get_modelmeta().custom_metadata_map or {}
        classes = json.loads(meta.get("classes_json", '["negative","neutral","positive"]'))

        proba_vec = outputs[1][0].tolist()
        proba_map = {classes[i]: float(proba_vec[i]) for i in range(min(len(classes), len(proba_vec)))}
        label = classes[int(max(range(len(proba_vec)), key=lambda i: proba_vec[i]))]

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
