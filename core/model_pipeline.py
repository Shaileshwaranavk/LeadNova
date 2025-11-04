import os
import time
import tempfile
from gradio_client import Client, handle_file

HF_SPACE_URL = os.getenv(
    "HYBRID_SPACE_URL",
    "https://shaileshwaran-lead-hybrid-analyzer.hf.space"
)

def analyze_leads_remote(product_name, description, features, labeled_file=None, new_file=None):
    """
    Sends uploaded CSVs and text inputs to Hugging Face Space.
    Handles both Django and Postman file inputs.
    """
    result = {"ml": {}, "ai": {}, "error": None}

    try:
        print(f"🔗 Connecting to Hugging Face Space: {HF_SPACE_URL}")
        client = Client(HF_SPACE_URL)
        print(f"✅ Client connected: {HF_SPACE_URL}")

        # --- Save Django InMemoryUploadedFile / TemporaryUploadedFile to disk
        def save_temp_file(uploaded_file, suffix=".csv"):
            if not uploaded_file:
                return None
            if hasattr(uploaded_file, "temporary_file_path"):
                return uploaded_file.temporary_file_path()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                for chunk in uploaded_file.chunks():
                    tmp.write(chunk)
                tmp.flush()
                return tmp.name

        labeled_path = save_temp_file(labeled_file)
        new_path = save_temp_file(new_file)

        if not new_path or not os.path.exists(new_path):
            raise ValueError("new_csv file is missing or not readable")

        labeled_handle = handle_file(labeled_path) if labeled_path else None
        new_handle = handle_file(new_path)

        # --- Call Hugging Face Space API
        for attempt in range(3):
            try:
                print(f"⚙️ Attempt {attempt + 1} → Remote /analyze API call...")
                response = client.predict(
                    product_name,
                    description,
                    features,
                    labeled_handle,
                    new_handle,
                    api_name="/predict"  # ✅ must match your Space
                )

                if isinstance(response, (list, tuple)):
                    result["ml"] = response[0] if len(response) > 0 else {}
                    result["ai"] = response[1] if len(response) > 1 else {}
                elif isinstance(response, dict):
                    result.update(response)
                else:
                    result["error"] = "Unexpected response type"
                break
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        else:
            result["error"] = "Remote inference failed after multiple retries."

    except Exception as e:
        print("❌ analyze_leads_remote error:", e)
        result["error"] = str(e)

    return result
