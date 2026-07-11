import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'datasets', 'processed_loan_dataset.csv')
DATASET_UNSCALED_PATH = os.path.join(BASE_DIR, 'datasets', 'processed_loan_dataset_unscaled.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
DB_URL = f"sqlite:///{os.path.join(BASE_DIR, 'loan_pricing.db')}"
