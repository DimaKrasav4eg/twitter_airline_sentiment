import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ml import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Tweets.csv")
    parser.add_argument("--out", required=True, help="Output path for model.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
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

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].values,
        df["airline_sentiment"].values,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["airline_sentiment"].values,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)

    metrics_path = out_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved model to: {out_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
