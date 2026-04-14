# HW4 MLOps — Olist Customer Satisfaction API

## Project Overview
This project deploys a machine learning model that predicts whether an Olist customer 
will leave a positive review (4-5 stars) based on order and delivery features available 
at the time of delivery. The model is a HistGradientBoostingClassifier trained on the 
Brazilian Olist e-commerce dataset and served via a REST API built with Flask.

## Live URL
https://hw4-mlops-awcg.onrender.com

## API Documentation

### GET /health
Returns the current status of the API and confirms the model is loaded.

**Response:**
```json
{"status": "healthy", "model": "loaded"}
```

---

### POST /predict
Accepts a single order record and returns a satisfaction prediction.

**Request body:**
```json
{
  "delivery_days": 12,
  "delivery_vs_estimated": 3,
  "late_delivery_flag": 1,
  "price": 149.99,
  "freight_value": 25.50,
  "freight_ratio": 0.17,
  "n_items": 1,
  "installments_max": 3,
  "product_category": "electronics",
  "seller_state": "SP",
  "payment_type": "credit_card"
}
```

**Response:**
```json
{"prediction": 1, "probability": 0.73, "label": "positive"}
```

---

### POST /predict/batch
Accepts a JSON array of up to 100 order records and returns predictions for all.

**Request body:** Array of objects using the same schema as /predict

**Response:** Array of prediction objects

---

## Input Schema

| Feature | Type | Description | Valid Values |
|---|---|---|---|
| delivery_days | float | Days from purchase to delivery | > 0 |
| delivery_vs_estimated | float | Days early/late vs estimate (negative = early) | any |
| late_delivery_flag | int | 1 if delivered after estimated date | 0 or 1 |
| price | float | Total order price in BRL | > 0 |
| freight_value | float | Total freight cost in BRL | > 0 |
| freight_ratio | float | freight_value / price | > 0 |
| n_items | int | Number of items in order | >= 1 |
| installments_max | int | Maximum payment installments | >= 1 |
| product_category | string | Product category in English | any valid Olist category |
| seller_state | string | Brazilian state code of seller | e.g. SP, RJ, MG |
| payment_type | string | Payment method | credit_card, boleto, voucher, debit_card |

---

## Local Setup

### Without Docker
```bash
git clone https://github.com/hernon33/hw4-mlops.git
cd hw4-mlops
pip install -r requirements.txt
python app.py
```
API will be available at http://127.0.0.1:5001

### With Docker
```bash
docker build -t hw4-api .
docker run -p 5001:5001 hw4-api
```
API will be available at http://127.0.0.1:5001

### Running Tests
```bash
python test_api.py
```

---

## Model Information

| Property | Value |
|---|---|
| Model type | HistGradientBoostingClassifier |
| Training data | Olist e-commerce dataset (~95,000 orders) |
| Accuracy | 0.8217 |
| F1 Score | 0.8971 |
| Precision | 0.8234 |
| Recall | 0.9855 |
| ROC-AUC | 0.7088 |

**Known limitations:** The model has high recall but moderate precision, meaning it 
tends to predict positive reviews more often than negative. It performs best on orders 
with complete delivery information and may be less reliable for edge cases such as 
very high-value orders or unusual product categories.
