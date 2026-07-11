import os
import numpy as np
import logging
from stable_baselines3 import PPO, DQN, SAC
from config.config import BASE_DIR

logger = logging.getLogger(__name__)

ACTION_MAP = {
    0: ("Reject", 0.0),
    1: ("Approve @ 8%", 8.0),
    2: ("Approve @ 12%", 12.0),
    3: ("Approve @ 16%", 16.0),
    4: ("Approve @ 20%", 20.0),
}

# ─── Model Registry (in-memory cache) ────────────────────────────
# Models are loaded once and cached here. After training completes,
# call reload_model(model_type) to pick up the new weights.

MODEL_PATHS = {
    "PPO": os.path.join(BASE_DIR, 'models', 'ppo_model.zip'),
    "DQN": os.path.join(BASE_DIR, 'models', 'dqn_model.zip'),
    "DDQN": os.path.join(BASE_DIR, 'models', 'ddqn_model.zip'),
    "SAC": os.path.join(BASE_DIR, 'models', 'sac_model.zip'),
}

MODEL_CLASSES = {
    "PPO": PPO,
    "DQN": DQN,
    "DDQN": DQN,   # DDQN uses SB3's DQN class
    "SAC": SAC,
}

# The actual loaded model objects live here
_model_cache: dict = {
    "PPO": None,
    "DQN": None,
    "DDQN": None,
    "SAC": None,
}


def _load_single_model(model_type: str):
    """Attempt to load a single model from disk. Returns the model or None."""
    path = MODEL_PATHS.get(model_type)
    cls = MODEL_CLASSES.get(model_type)
    if path and cls and os.path.exists(path):
        try:
            model = cls.load(path)
            logger.info(f"✅ Loaded {model_type} model from {path}")
            return model
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {model_type}: {e}")
    return None


def load_all_models():
    """Scan disk and load every available model into the cache.
    Called once at application startup."""
    for model_type in MODEL_PATHS:
        _model_cache[model_type] = _load_single_model(model_type)
    loaded = [k for k, v in _model_cache.items() if v is not None]
    logger.info(f"Model registry initialised. Loaded: {loaded or 'none'}")


def reload_model(model_type: str):
    """Reload a specific model after training has finished."""
    if model_type not in MODEL_PATHS:
        logger.warning(f"Unknown model type: {model_type}")
        return False
    _model_cache[model_type] = _load_single_model(model_type)
    return _model_cache[model_type] is not None


def get_model_status() -> dict:
    """Return the status of every model (loaded / not-trained)."""
    status = {}
    for model_type in MODEL_PATHS:
        path = MODEL_PATHS[model_type]
        if _model_cache[model_type] is not None:
            status[model_type] = {
                "status": "loaded",
                "file": os.path.basename(path),
                "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2) if os.path.exists(path) else None,
            }
        elif os.path.exists(path):
            # File exists on disk but not yet loaded into memory
            status[model_type] = {
                "status": "on_disk",
                "file": os.path.basename(path),
                "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
            }
        else:
            status[model_type] = {
                "status": "not_trained",
                "file": None,
                "size_mb": None,
            }
    return status


# ─── Prediction Logic ─────────────────────────────────────────────

def predict_all_models(state_vector: np.ndarray, credit_score: float, income: float) -> dict:
    """Run predictions using cached models. Falls back to heuristics
    for any model that is not loaded."""

    risk_score = round(max(0.0, min(1.0, 1.0 - (credit_score * 0.6 + min(income / 50000, 1.0) * 0.4))), 4)

    results = {}
    model_rewards = {}

    # PPO
    ppo_model = _model_cache.get("PPO")
    if ppo_model:
        action, _ = ppo_model.predict(state_vector, deterministic=True)
        action = int(action)
        decision, rate = ACTION_MAP.get(action, ("Reject", 0.0))
        results['ppo_prediction'] = f"{decision}"
        results['ppo_rate'] = rate
        model_rewards['PPO'] = rate * (1 - risk_score) if rate > 0 else -1
    else:
        results['ppo_prediction'] = "Approve @ 12%" if risk_score < 0.4 else "Reject"
        results['ppo_rate'] = 12.0 if risk_score < 0.4 else 0.0
        model_rewards['PPO'] = results['ppo_rate'] * (1 - risk_score)

    # DQN
    dqn_model = _model_cache.get("DQN")
    if dqn_model:
        action, _ = dqn_model.predict(state_vector, deterministic=True)
        action = int(action)
        decision, rate = ACTION_MAP.get(action, ("Reject", 0.0))
        results['dqn_prediction'] = f"{decision}"
        results['dqn_rate'] = rate
        model_rewards['DQN'] = rate * (1 - risk_score) if rate > 0 else -1
    else:
        results['dqn_prediction'] = "Approve @ 10%" if risk_score < 0.35 else "Reject"
        results['dqn_rate'] = 10.0 if risk_score < 0.35 else 0.0
        model_rewards['DQN'] = results['dqn_rate'] * (1 - risk_score)

    # DDQN
    ddqn_model = _model_cache.get("DDQN")
    if ddqn_model:
        action, _ = ddqn_model.predict(state_vector, deterministic=True)
        action = int(action)
        decision, rate = ACTION_MAP.get(action, ("Reject", 0.0))
        results['ddqn_prediction'] = f"{decision}"
        results['ddqn_rate'] = rate
        model_rewards['DDQN'] = rate * (1 - risk_score) if rate > 0 else -1
    else:
        results['ddqn_prediction'] = "Approve @ 11%" if risk_score < 0.38 else "Reject"
        results['ddqn_rate'] = 11.0 if risk_score < 0.38 else 0.0
        model_rewards['DDQN'] = results['ddqn_rate'] * (1 - risk_score)

    # SAC
    sac_model = _model_cache.get("SAC")
    if sac_model:
        action, _ = sac_model.predict(state_vector, deterministic=True)
        val = float(action[0])
        if val < 0.1:
            results['sac_prediction'] = "Reject"
            results['sac_rate'] = 0.0
            model_rewards['SAC'] = -1
        else:
            rate = round(5.0 + ((val - 0.1) / 0.9) * 20.0, 2)
            results['sac_prediction'] = f"Approve @ {rate}%"
            results['sac_rate'] = rate
            model_rewards['SAC'] = rate * (1 - risk_score)
    else:
        results['sac_prediction'] = "Approve @ 11.5%" if risk_score < 0.42 else "Reject"
        results['sac_rate'] = 11.5 if risk_score < 0.42 else 0.0
        model_rewards['SAC'] = results['sac_rate'] * (1 - risk_score)

    # Best model
    best_model = max(model_rewards, key=model_rewards.get) if model_rewards else "PPO"
    best_rate = results.get(f"{best_model.lower()}_rate", 12.0)

    confidence = round(max(0.5, 1.0 - risk_score * 0.5), 4)

    return {
        **results,
        'best_model': best_model,
        'recommended_interest_rate': best_rate if best_rate > 0 else None,
        'risk_score': risk_score,
        'confidence': confidence,
    }

