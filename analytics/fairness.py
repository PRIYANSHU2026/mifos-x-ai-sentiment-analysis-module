"""
Fairness Auditing Module
========================
Calculates Disparate Impact Ratio (DIR) and Equal Opportunity Difference (EOD)
for sensitive features (gender, region) across all RL models.

Regulatory reference: The 4/5ths (80%) rule — if DIR < 0.8, the model exhibits
illegal disparate impact and must be flagged for retraining.
"""

import pandas as pd
import numpy as np
import logging
from sqlalchemy.orm import Session
from database.database import SessionLocal
from database import models

logger = logging.getLogger(__name__)

# ─── Threshold ────────────────────────────────────────────────────
DISPARATE_IMPACT_THRESHOLD = 0.8

# ─── In-memory flag registry ─────────────────────────────────────
# Maps model_name -> {"flagged": bool, "reason": str, "details": dict}
flagged_models: dict = {}


def _build_joined_dataframe(db: Session) -> pd.DataFrame:
    """
    Join LoanApplication + RLPrediction into a single pandas DataFrame.
    Each row represents one application with its RL prediction results.
    """
    rows = (
        db.query(
            models.LoanApplication.id.label("app_id"),
            models.LoanApplication.gender,
            models.LoanApplication.region,
            models.LoanApplication.credit_score,
            models.LoanApplication.income,
            models.LoanApplication.age,
            models.RLPrediction.recommended_interest_rate,
            models.RLPrediction.risk_score,
            models.RLPrediction.best_model,
            models.RLPrediction.ppo_prediction,
            models.RLPrediction.dqn_prediction,
            models.RLPrediction.ddqn_prediction,
        )
        .join(
            models.RLPrediction,
            models.RLPrediction.application_id == models.LoanApplication.id,
        )
        .all()
    )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "app_id", "gender", "region", "credit_score", "income", "age",
        "recommended_interest_rate", "risk_score", "best_model",
        "ppo_prediction", "dqn_prediction", "ddqn_prediction",
    ])
    return df


def _disparate_impact(df: pd.DataFrame, feature: str,
                      privileged: str, unprivileged: str,
                      favorable_col: str) -> float:
    """
    Calculate Disparate Impact Ratio.
    DIR = P(favorable | unprivileged) / P(favorable | privileged)
    """
    priv = df[df[feature] == privileged]
    unpriv = df[df[feature] == unprivileged]

    if len(priv) == 0 or len(unpriv) == 0:
        return float("nan")

    rate_priv = priv[favorable_col].mean()
    rate_unpriv = unpriv[favorable_col].mean()

    if rate_priv == 0:
        return float("nan")

    return round(rate_unpriv / rate_priv, 4)


def _equal_opportunity_difference(df: pd.DataFrame, feature: str,
                                  privileged: str, unprivileged: str,
                                  favorable_col: str) -> float:
    """
    Calculate Equal Opportunity Difference.
    EOD = TPR_unprivileged - TPR_privileged
    Here we approximate TPR as the favorable rate (since we don't have
    ground-truth labels, we use the model's own favorable decisions).
    A value of 0 means perfect equality; negative means bias against unprivileged.
    """
    priv = df[df[feature] == privileged]
    unpriv = df[df[feature] == unprivileged]

    if len(priv) == 0 or len(unpriv) == 0:
        return float("nan")

    rate_priv = priv[favorable_col].mean()
    rate_unpriv = unpriv[favorable_col].mean()

    return round(rate_unpriv - rate_priv, 4)


def run_fairness_audit(db: Session = None) -> dict:
    """
    Main audit entry point. Computes fairness metrics for all models
    across gender and region features.

    Returns a dict with:
      - "summary": high-level metrics
      - "by_model": per-model breakdown
      - "flagged_models": list of models with DIR < 0.8
      - "group_rates": favorable rates by group for charting
    """
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        df = _build_joined_dataframe(db)

        if df.empty:
            return {
                "summary": {},
                "by_model": [],
                "flagged_models": [],
                "group_rates": {},
                "total_applications": 0,
            }

        # Define "favorable" = interest rate below the overall median
        median_rate = df["recommended_interest_rate"].median()
        df["favorable"] = (df["recommended_interest_rate"] <= median_rate).astype(int)

        # Sensitive feature definitions
        audit_configs = [
            {"feature": "gender", "privileged": "Male", "unprivileged": "Female"},
            {"feature": "region", "privileged": "Urban", "unprivileged": "Rural"},
            {"feature": "region", "privileged": "Urban", "unprivileged": "Semiurban"},
        ]

        # ─── Overall metrics ─────────────────────────────────────
        summary = {}
        for cfg in audit_configs:
            key = f"{cfg['feature']}_{cfg['unprivileged'].lower()}"
            di = _disparate_impact(df, cfg["feature"], cfg["privileged"],
                                   cfg["unprivileged"], "favorable")
            eod = _equal_opportunity_difference(df, cfg["feature"],
                                                cfg["privileged"],
                                                cfg["unprivileged"], "favorable")
            summary[key] = {
                "disparate_impact": di,
                "equal_opportunity_diff": eod,
                "privileged_group": cfg["privileged"],
                "unprivileged_group": cfg["unprivileged"],
                "feature": cfg["feature"],
            }

        # ─── Per-model breakdown ──────────────────────────────────
        by_model = []
        new_flags = []

        model_names = df["best_model"].dropna().unique().tolist()
        for model_name in model_names:
            model_df = df[df["best_model"] == model_name]
            if len(model_df) < 2:
                continue

            # Recompute favorable based on model-specific median
            model_median = model_df["recommended_interest_rate"].median()
            model_df = model_df.copy()
            model_df["favorable"] = (
                model_df["recommended_interest_rate"] <= model_median
            ).astype(int)

            for cfg in audit_configs:
                di = _disparate_impact(model_df, cfg["feature"],
                                       cfg["privileged"], cfg["unprivileged"],
                                       "favorable")
                eod = _equal_opportunity_difference(model_df, cfg["feature"],
                                                    cfg["privileged"],
                                                    cfg["unprivileged"],
                                                    "favorable")
                is_flagged = (not np.isnan(di)) and di < DISPARATE_IMPACT_THRESHOLD

                entry = {
                    "model": model_name,
                    "feature": cfg["feature"],
                    "privileged_group": cfg["privileged"],
                    "unprivileged_group": cfg["unprivileged"],
                    "disparate_impact": di if not np.isnan(di) else None,
                    "equal_opportunity_diff": eod if not np.isnan(eod) else None,
                    "sample_size_privileged": int(
                        len(model_df[model_df[cfg["feature"]] == cfg["privileged"]])
                    ),
                    "sample_size_unprivileged": int(
                        len(model_df[model_df[cfg["feature"]] == cfg["unprivileged"]])
                    ),
                    "flagged": is_flagged,
                }
                by_model.append(entry)

                if is_flagged:
                    new_flags.append(entry)
                    # Persist flag in memory
                    flagged_models[model_name] = {
                        "flagged": True,
                        "reason": (
                            f"Disparate Impact ({di:.2f}) below threshold "
                            f"({DISPARATE_IMPACT_THRESHOLD}) for "
                            f"{cfg['feature']}: {cfg['unprivileged']}"
                        ),
                        "details": entry,
                    }
                    logger.warning(
                        f"⚠️ FAIRNESS ALERT: Model '{model_name}' flagged — "
                        f"DIR={di:.4f} for {cfg['feature']} "
                        f"({cfg['unprivileged']} vs {cfg['privileged']}). "
                        f"Model marked as 'Needs Retraining'."
                    )

        # ─── Group rates for charting ─────────────────────────────
        group_rates = {}
        for cfg in audit_configs:
            key = f"{cfg['feature']}_{cfg['unprivileged'].lower()}"
            priv = df[df[cfg["feature"]] == cfg["privileged"]]
            unpriv = df[df[cfg["feature"]] == cfg["unprivileged"]]
            group_rates[key] = {
                "privileged": {
                    "group": cfg["privileged"],
                    "favorable_rate": round(priv["favorable"].mean(), 4) if len(priv) > 0 else None,
                    "avg_interest_rate": round(priv["recommended_interest_rate"].mean(), 4) if len(priv) > 0 else None,
                    "avg_risk_score": round(priv["risk_score"].mean(), 4) if len(priv) > 0 else None,
                    "count": int(len(priv)),
                },
                "unprivileged": {
                    "group": cfg["unprivileged"],
                    "favorable_rate": round(unpriv["favorable"].mean(), 4) if len(unpriv) > 0 else None,
                    "avg_interest_rate": round(unpriv["recommended_interest_rate"].mean(), 4) if len(unpriv) > 0 else None,
                    "avg_risk_score": round(unpriv["risk_score"].mean(), 4) if len(unpriv) > 0 else None,
                    "count": int(len(unpriv)),
                },
            }

        # ─── Persist flagged entries to DB ────────────────────────
        for flag_entry in new_flags:
            audit_log = models.FairnessAuditLog(
                model_name=flag_entry["model"],
                feature=flag_entry["feature"],
                group_a=flag_entry["privileged_group"],
                group_b=flag_entry["unprivileged_group"],
                disparate_impact=flag_entry["disparate_impact"],
                equal_opportunity_diff=flag_entry["equal_opportunity_diff"],
                flagged=True,
            )
            db.add(audit_log)

        # Also persist non-flagged entries for history
        for entry in by_model:
            if not entry["flagged"]:
                audit_log = models.FairnessAuditLog(
                    model_name=entry["model"],
                    feature=entry["feature"],
                    group_a=entry["privileged_group"],
                    group_b=entry["unprivileged_group"],
                    disparate_impact=entry["disparate_impact"],
                    equal_opportunity_diff=entry["equal_opportunity_diff"],
                    flagged=False,
                )
                db.add(audit_log)

        db.commit()

        return {
            "summary": summary,
            "by_model": by_model,
            "flagged_models": [
                {"model": k, **v} for k, v in flagged_models.items()
            ],
            "group_rates": group_rates,
            "total_applications": int(len(df)),
        }

    finally:
        if close_db:
            db.close()


def get_flagged_models() -> dict:
    """Return the current in-memory flagged models registry."""
    return dict(flagged_models)


def is_model_flagged(model_name: str) -> bool:
    """Check if a specific model is currently flagged."""
    return model_name in flagged_models and flagged_models[model_name].get("flagged", False)
