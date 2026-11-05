# Heart Disease Prediction API

FastAPI service that serves a scikit-learn classifier trained on the
[Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset),
packaged with Docker and deployable on Render.

**Live URL:** https://heart-disease-api-fastapi-docker.onrender.com

**Swagger UI:** https://heart-disease-api-fastapi-docker.onrender.com/docs

## Project structure

```
app/
  main.py        FastAPI app and endpoints
  schemas.py     Pydantic models (input features, responses)
model/
  heart_model.joblib   trained model (created by train.py)
train.py         training script
Dockerfile
docker-compose.yml
requirements.txt
```

## 1. Train the model

1. Download `heart.csv` from Kaggle and place it at `data/heart.csv`.
2. Install dependencies and train:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

This writes `model/heart_model.joblib` (a Random Forest plus its feature list).
Commit the model file so Docker and Render can copy it into the image.

## 2. Run with Docker

```bash
docker-compose build
docker-compose up
```

Open http://localhost:8000/docs for Swagger UI.

## 3. Endpoints

| Method | Path       | Description                                |
|--------|------------|--------------------------------------------|
| GET    | `/health`  | Liveness check                             |
| GET    | `/info`    | Model type and feature list                |
| POST   | `/predict` | Returns `heart_disease: true/false` + probability |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,
       "thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

Response:

```json
{"heart_disease": true, "probability": 0.87}
```

## 4. Deploy to Render

1. Push this repo to GitHub (including `model/heart_model.joblib`).
2. On Render: **New > Web Service**, connect the repo.
3. Set **Language/Environment** to **Docker**. Leave the root directory empty
   (the Dockerfile is at the project root).
4. Create the service and wait for the build. Render provides `$PORT`, which the
   Dockerfile's `CMD` already uses.
5. Test the live URL:

```bash
curl https://<your-service>.onrender.com/health
```

Note: Render's free tier sleeps after inactivity, so the first request may be slow.

## Disclaimer

Educational project only. Not a medical device and not for clinical use.
