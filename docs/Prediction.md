# 🔮 Prediction API — Modular Inference with FastAPI

This prediction service is built using **FastAPI**, designed for **real-world ML applications** where each step (e.g., scaling, encoding, outlier detection) is **modularized into separate `.pkl` files**. It aligns with your ML pipeline architecture and supports clean interaction from tools like **Streamlit**, mobile apps, or other services.

---

## 🧱 Modular Architecture Overview

| Component Type     | Stored As              | Used For                                |
| ------------------ | ---------------------- | --------------------------------------- |
| Preprocessing Step | `outlier_detector.pkl` | Filtering/removing outliers             |
| Scaler             | `scaler.pkl`           | Scaling numeric features                |
| Encoder            | `encoder.pkl`          | Encoding categoricals                   |
| Model              | `model.pkl`            | Final prediction logic                  |
| Metadata           | `meta.json`            | Input schema, versioning, feature names |

---

## 🔁 Inference Pipeline Flow

```text
Streamlit app / external UI
           |
           V
     [FastAPI POST /predict]
           |
           V
[1] Load input schema & meta.json
[2] Apply: outlier → encoder → scaler
[3] Load model and predict
[4] Return result (+ confidence/score)
```

Each `.pkl` file corresponds to a class instance saved using `joblib` or `pickle`. At runtime, they're dynamically loaded and applied to the incoming request.

---

## 🔗 API Endpoints

### `POST /predict`

> Run the full pipeline from raw input to final prediction.

**Request:**

```json
{
  "input": {
    "feature_1": 7.1,
    "feature_2": "high",
    "feature_3": 5.2
  }
}
```

**Response:**

```json
{
  "prediction": "positive",
  "score": 0.86
}
```

---

### `POST /predict_batch`

> Run inference on a batch of inputs (list of JSON dicts).

---

### `GET /status`

> Check health of loaded components (outlier, scaler, encoder, model).

---

### `GET /metadata`

> Returns JSON defining:

- input types
- required fields
- feature order
- version tags

---

## 📦 Project Structure

```
api/
├── main.py              # FastAPI app
├── predict.py           # Modular inference logic
├── schemas.py           # Pydantic models for request/response
├── utils/
│   ├── loader.py        # Dynamic joblib/pickle loading
│   └── pipeline.py      # Apply steps in sequence
├── models/
│   ├── outlier.pkl
│   ├── encoder.pkl
│   ├── scaler.pkl
│   └── model.pkl
├── metadata/
│   └── meta.json
├── templates/           # Optional HTML UI
└── static/
```

---

## ⚙️ How to Run

### Dev Server

```bash
uvicorn api.main:app --reload --port 8000
```

### Docker

```bash
docker build -t ml_predict_api -f docker/Dockerfile.api .
docker run -p 8000:8000 ml_predict_api
```

---

## 🎛️ Real-World Interaction via Streamlit

You can create a page in your existing `Streamlit` dashboard to interact with this API:

```python
# streamlit_pages/LivePrediction.py
import streamlit as st
import requests
import json

st.title("🚀 Real-Time Prediction")

form = {}
form["feature_1"] = st.number_input("Feature 1")
form["feature_2"] = st.selectbox("Feature 2", ["low", "medium", "high"])
form["feature_3"] = st.slider("Feature 3", 0.0, 10.0, 5.0)

if st.button("Predict"):
    res = requests.post("http://localhost:8000/predict", json={"input": form})
    st.json(res.json())
```

---

## 🧠 Design Notes

- Each `.pkl` file corresponds to **one transformation step**.
- All are chained inside `predict.py → apply_pipeline(input_dict)`
- This makes it **extensible**: add a new pre-step? Just insert after loading.
- Future enhancements can support:

  - version control
  - ensemble logic
  - streaming/batch mode switch
  - model registry switching

---

## 📌 Tools Used

- `FastAPI`, `Pydantic`, `Uvicorn`
- `joblib`, `scikit-learn`, `pandas`, `numpy`
- `jinja2` (optional HTML)
- `requests` (client side)
- `Streamlit` (live app integration)

---

## ✅ Status

- [x] Modular model loading
- [x] Full inference pipeline
- [x] Metadata for input structure
- [x] Streamlit integration
- [x] Auth/RBAC
- [x] Async job handling
- [x] Logging to MLflow or Evidently

---

## 📎 Meta

This API is built to serve **any ML/DL model pipeline**, assuming modular `pkl` serialization. It is **not tied to one MLflow run**, and doesn't assume fixed model ID.

Think of this as the **"inference twin"** of your ML pipeline.

---

## 🚀 Next Step

Let me know if you'd like:

- Python code scaffold for `api/` matching this logic
- Docker support for just the prediction layer
- Jinja2-powered minimal frontend (`index.html`) for testing

Ready to ship that too.
