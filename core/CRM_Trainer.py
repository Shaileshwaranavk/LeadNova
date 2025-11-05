import requests
import pandas as pd
from django.conf import settings
from .models import Customer, CustomerReview, ModelFeedback  # ✅ fixed import
import logging

logger = logging.getLogger(__name__)

HF_LEAD_ANALYZER_URL = "https://huggingface.co/spaces/Shaileshwaran/lead-hybrid-analyzer"
HF_SALES_PITCH_URL = "https://huggingface.co/spaces/Shaileshwaran/sales-pitch-infer"


def analyze_leads_from_db():
    """Send customer data to Hugging Face Lead Analyzer"""
    customers = list(Customer.objects.all().values())
    if not customers:
        return {"status": "No customer data to analyze"}

    df = pd.DataFrame(customers)
    payload = {"inputs": df.to_dict(orient="records")}

    try:
        res = requests.post(HF_LEAD_ANALYZER_URL, json=payload, timeout=120)
        if res.status_code == 200:
            result = res.json()
            logger.info("Lead analysis successful via Hugging Face.")
            return result
        else:
            logger.error(f"Lead Analyzer failed: {res.status_code}")
            return {"error": "Lead Analyzer request failed", "status_code": res.status_code}
    except Exception as e:
        logger.exception("Error calling HF Lead Analyzer")
        return {"error": str(e)}


def generate_sales_pitch(product_name: str, description: str, features: str):
    """Call Hugging Face Sales Pitch Generator"""
    payload = {
        "inputs": {
            "product_name": product_name,
            "description": description,
            "features": features
        }
    }

    try:
        res = requests.post(HF_SALES_PITCH_URL, json=payload, timeout=120)
        if res.status_code == 200:
            return res.json()
        else:
            return {"error": "Sales pitch API failed", "status_code": res.status_code}
    except Exception as e:
        logger.exception("Error calling HF Sales Pitch Infer")
        return {"error": str(e)}


def retrain_from_feedback():
    """
    Placeholder for retraining logic. Could trigger retraining pipeline on HF or local model.
    """
    feedbacks = list(ModelFeedback.objects.all().values("model_name", "feedback_text", "rating"))
    if not feedbacks:
        return {"status": "No feedbacks to retrain on"}
    # You can later integrate with a Hugging Face repo push or API trigger
    return {"status": "Feedback collected for retraining", "count": len(feedbacks)}
