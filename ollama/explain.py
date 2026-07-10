import requests
import json
import logging

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma2:2b"

def generate_explanation(applicant_data: dict, prediction_data: dict, model=DEFAULT_MODEL) -> dict:
    """
    Calls local Ollama API to generate human-readable explanations of the RL decision.
    """
    prompt = f"""
    You are an expert AI Banking Assistant. The Reinforcement Learning (RL) ensemble has made a loan decision.
    Your job is ONLY to explain this decision clearly. NEVER change the interest rate or decision.

    Applicant Data:
    {json.dumps(applicant_data, indent=2)}

    RL Engine Prediction:
    {json.dumps(prediction_data, indent=2)}

    Please provide a JSON response with exactly these keys:
    1. "customer_friendly_explanation": A polite, simple explanation for the customer.
    2. "officer_technical_explanation": A detailed technical explanation for the loan officer justifying the risk vs reward.
    3. "suggested_improvements": What the applicant can do to get a better rate next time.
    
    Return ONLY valid JSON.
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "{}")
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from Ollama")
                return {
                    "customer_friendly_explanation": "Error parsing AI response.",
                    "officer_technical_explanation": response_text,
                    "suggested_improvements": "N/A"
                }
        else:
            logger.error(f"Ollama API Error: {response.text}")
            return {"error": "Ollama API returned an error."}
    except requests.exceptions.ConnectionError:
        logger.warning("Ollama is not running locally on port 11434.")
        # Return fallback mock data if Ollama is not installed/running
        return {
            "customer_friendly_explanation": "The AI has analyzed your application. Based on your credit score and current debt, we have offered you a personalized interest rate.",
            "officer_technical_explanation": "[OLLAMA OFFLINE] RL Model optimized the rate balancing the applicant's credit score against the predicted default probability. The SAC agent selected this continuous rate for maximum expected reward.",
            "suggested_improvements": "Lower your existing debt ratio and improve your credit score."
        }
    except Exception as e:
        logger.error(f"Error calling Ollama: {e}")
        return {"error": str(e)}
