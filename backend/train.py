import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import onnx
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ml import build_pipeline, normalize_text


def get_commit_hash() -> str:
    env_hash = os.getenv("GIT_COMMIT")
    if env_hash:
        return env_hash
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def add_onnx_metadata(model_onnx: onnx.ModelProto, experiment_name: str, classes) -> onnx.ModelProto:
    meta = {p.key: p.value for p in model_onnx.metadata_props}

    commit_hash = get_commit_hash()
    saved_at = datetime.now(timezone.utc).isoformat()

    meta.update(
        {
            "commit_hash": commit_hash,
            "saved_at": saved_at,
            "experiment_name": experiment_name,
            "classes": str(classes),
        }
    )

    del model_onnx.metadata_props[:]
    for k, v in meta.items():
        p = model_onnx.metadata_props.add()
        p.key = str(k)
        p.value = str(v)

    return model_onnx

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Tweets.csv")
    parser.add_argument("--out", required=True, help="Output path for model.onnx")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", default="baseline", help="Experiment name for ONNX metadata")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    required_cols = {"text", "airline_sentiment"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"CSV must contain columns {sorted(required_cols)}; got {sorted(df.columns)}"
        )

    df = df.dropna(subset=["text", "airline_sentiment"]).copy()
    df["text"] = df["text"].astype(str)
    df["airline_sentiment"] = df["airline_sentiment"].astype(str)

    X = df["text"].astype(str).map(normalize_text).values
    y = df["airline_sentiment"].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    classes = list(model.named_steps["clf"].classes_)


    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    initial_types = [("input", StringTensorType([None, 1]))]

    options = {id(model.named_steps["clf"]): {"zipmap": False}}

    model_onnx = convert_sklearn(
        model,
        initial_types=initial_types,
        options=options,
        target_opset=12,
    )

    model_onnx = add_onnx_metadata(
        model_onnx, 
        experiment_name=args.experiment_name,
        classes=classes,
    )
    onnx.save_model(model_onnx, out_path)

    metrics_path = out_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved ONNX model to: {out_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
