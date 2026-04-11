import requests

BASE_URL = "http://127.0.0.1:5001"

def run_test(test_name, response, expected_status):
    if response.status_code == expected_status:
        print(f"PASS - {test_name}")
        print(f"       Response: {response.json()}\n")
    else:
        print(f"FAIL - {test_name}")
        print(f"       Expected status {expected_status}, got {response.status_code}")
        print(f"       Response: {response.json()}\n")


# test 1: health check
r = requests.get(f"{BASE_URL}/health")
run_test("GET /health", r, 200)

# test 2: valid single prediction
payload = {
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
    "payment_type": "credit_card",
}
r = requests.post(f"{BASE_URL}/predict", json=payload)
run_test("POST /predict - valid input", r, 200)

# test 3: valid batch of 5 records
batch = [payload for _ in range(5)]
r = requests.post(f"{BASE_URL}/predict/batch", json=batch)
run_test("POST /predict/batch - 5 records", r, 200)
assert len(r.json()) == 5, "expected 5 predictions"
print(f"       Confirmed: {len(r.json())} predictions returned\n")

# test 4: missing required field
bad_payload = {k: v for k, v in payload.items() if k != "delivery_days"}
r = requests.post(f"{BASE_URL}/predict", json=bad_payload)
run_test("POST /predict - missing field", r, 400)

# test 5: invalid type (string for price)
bad_type = payload.copy()
bad_type["price"] = "not_a_number"
r = requests.post(f"{BASE_URL}/predict", json=bad_type)
run_test("POST /predict - invalid type", r, 400)