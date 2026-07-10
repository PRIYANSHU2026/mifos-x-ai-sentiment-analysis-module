import requests
import random
import time

API_URL = "http://127.0.0.1:8000/api"

print("Populating dummy data...")

# 1. Create some loan applications
purposes = ["Personal", "Business", "Education", "Home"]
regions = ["Urban", "Semiurban", "Rural"]
employments = ["Salaried", "Self-Employed", "Unemployed"]
genders = ["Male", "Female", "Other"]

for i in range(10):
    app_data = {
        "age": random.randint(21, 60),
        "gender": random.choice(genders),
        "employment": random.choice(employments),
        "income": random.uniform(3000, 15000),
        "credit_score": random.uniform(0.3, 0.95),
        "loan_amount": random.uniform(5000, 50000),
        "loan_purpose": random.choice(purposes),
        "existing_debt": random.uniform(0, 10000),
        "loan_tenure": random.choice([12, 24, 36, 48, 60]),
        "repayment_history": random.uniform(0.5, 1.0),
        "region": random.choice(regions)
    }
    
    try:
        res = requests.post(f"{API_URL}/predict", json=app_data)
        if res.status_code == 200:
            result = res.json()
            app_id = result.get("application_id")
            
            # Approve some, reject some
            if random.random() > 0.3:
                decision_data = {
                    "application_id": app_id,
                    "officer_name": "Admin System",
                    "remarks": "Looks good based on AI recommendation.",
                    "approved": True,
                    "rejected": False
                }
            else:
                decision_data = {
                    "application_id": app_id,
                    "officer_name": "Admin System",
                    "remarks": "Too risky.",
                    "approved": False,
                    "rejected": True
                }
                
            requests.post(f"{API_URL}/loan-decisions", json=decision_data)
            print(f"Processed Application {app_id}")
    except Exception as e:
        print(f"Error: {e}")

print("Data population complete!")
