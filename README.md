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
│   │   ├── model.onnx
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
├── monitoring
│   └── prometheus.yml
└── run.sh
```

## Model training

### Quick start

```bash
chmod +x run.sh
./run.sh
```

After the run:
- Frontend: http://localhost:8080
- Backend: http://localhost:8000
- Prometheus: http://localhost:9090

## Using the API
```bash
curl -s -X POST http://localhost:8000/forward \
  -H "Content-Type: application/json" \
  -d '{"text":"Thanks you, great service!"}'
```
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
### GET /metadata
```bash
curl -s http://localhost:8000/metadata
```
```json
{
    "commit_hash":"<hash>",
    "saved_at":"<timestamp>",
    "experiment_name":"v1"
}
```
### GET /metrics
```bash
curl -s http://localhost:8000/metadata
```
```
<Prometheus-metrics>
```
