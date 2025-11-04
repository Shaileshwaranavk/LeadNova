import os
from gradio_client import Client

# ==============================================================
# 🌐 HUGGING FACE SPACE INFERENCE HELPER
# ==============================================================

# Your Space endpoint — replace if you rename it later
INFER_API_URL = os.getenv(
    "INFER_API_URL",
    "https://shaileshwaran-sales-pitch-infer.hf.space"
)

def generate_remote_recommendation(product_name, description, features):
    """
    Calls the Hugging Face Space (Gradio app) using Gradio Client.
    """
    try:
        # Connect to the Space
        client = Client(INFER_API_URL)

        # Call the Gradio function (matches your app.py inputs order)
        result = client.predict(
            product_name,
            description,
            features,
            api_name="/predict"  # Default API route created by Gradio Interface
        )

        # Gradio returns a list [sales_pitch, llm_insight, highlighted_features, feature_weightage]
        return {
            "sales_pitch": result[0] if len(result) > 0 else "",
            "llm_insight": result[1] if len(result) > 1 else "",
            "highlighted_features": result[2] if len(result) > 2 else "",
            "feature_weightage": result[3] if len(result) > 3 else "",
        }

    except Exception as e:
        print("❌ Remote inference error:", e)
        return {
            "error": str(e),
            "sales_pitch": "",
            "llm_insight": "",
            "highlighted_features": "",
            "feature_weightage": ""
        }
