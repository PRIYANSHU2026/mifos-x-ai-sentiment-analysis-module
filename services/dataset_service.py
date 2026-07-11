import pandas as pd
import json
import os
import logging

logger = logging.getLogger(__name__)

# The features our current RL environment requires
REQUIRED_FEATURES = [
    "age", "gender", "employment", "income", "credit_score", "loan_amount", 
    "loan_purpose", "existing_debt", "loan_tenure", "repayment_history", "region",
    "collateral", "existing_loans", "education", "decision" # decision is the target
]

def load_and_preprocess_dataset(filepath: str, schema_mapping: str):
    """
    Loads a dataset from CSV or Excel, renames columns according to schema_mapping,
    validates required columns, and cleans the data.
    """
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel.")

    # Apply mapping
    mapping = json.loads(schema_mapping) if isinstance(schema_mapping, str) else schema_mapping
    df = df.rename(columns=mapping)

    # Check for missing required columns
    missing = [f for f in REQUIRED_FEATURES if f not in df.columns and f != "decision"] # decision can be optional if just predicting
    if missing:
        logger.warning(f"Dataset is missing required features: {missing}. Creating dummies.")
        for col in missing:
            df[col] = 0 # Default fallback, though UI should warn user

    # Handle basic missing values in the dataframe
    # Fill numerical with median, categorical with mode
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
            
    # Normalize categorical to strings
    for col in ["gender", "employment", "loan_purpose", "region", "collateral", "education"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # The RL env handles the categorical encoding natively via LabelEncoder.
    # So we just return the cleaned dataset path for the env to use.
    
    output_path = filepath.replace(".csv", "_processed.csv").replace(".xlsx", "_processed.csv")
    df.to_csv(output_path, index=False)
    
    return output_path
