# Twitter Airline Sentiment — ML service

## Dataset
The dataset is being used: https://www.kaggle.com/crowdflower/twitter-airline-sentiment

The CSV file `Tweets.csv` is expected (columns `text` and `airline_sentiment').

## Project structure

```
.
├── README.md
├── backend
│   ├── Dockerfile
│   ├── app.py
│   ├── artifacts
│   │   ├── model.joblib
│   │   └── model.metrics.json
│   ├── ml.py
│   ├── requirements.txt
│   └── train.py
├── data
│   └── Tweets.csv
├── docker-compose.yml
├── frontend
│   ├── Dockerfile
│   ├── index.html
│   └── nginx.conf
└── monitoring
    └── prometheus.yml
```

## Model training

### Option 1: Train on host
```bash
python -m venv .venv
source .venv/bin/activate

pip install -r backend/requirements.txt
python backend/train.py --data data/Tweets.csv --out backend/artifacts/model.joblib
```
### Option 2: Learning via Docker

```bash
docker compose run --rm backend python train.py --data /data/Tweets.csv --out /artifacts/model.joblib
```


After the run:
- Frontend: http://localhost:8080
- Backend: http://localhost:8000
- Prometheus: http://localhost:9090

## Using the API

Input example:
```bash
curl -s -X POST http://localhost:8000/forward \
  -H "Content-Type: application/json" \
  -d '{"text":"Thanks you, great service!"}'
```
Output example:
```json
{
    "label": "positive",
    "proba": {
        "negative": 0.0009009299826494685,
        "neutral":0.004538340991177582,
        "positive":0.994560729026173
    }
}
```
