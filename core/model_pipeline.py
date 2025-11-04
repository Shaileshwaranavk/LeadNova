import os
import time
import tempfile
from gradio_client import Client, handle_file

HF_SPACE_URL = os.getenv(
    "HYBRID_SPACE_URL",
    "https://shaileshwaran-lead-hybrid-analyzer.hf.space"
)

def analyze_leads_remote(product_name, description, features, labeled_file=None, new_file=None):
    result = {"ml": {}, "ai": {}}

    try:
        print(f"🔗 Connecting to Hugging Face Space: {HF_SPACE_URL}")
        client = Client(HF_SPACE_URL)
        print(f"Loaded as API: {HF_SPACE_URL}/ ✔")
        print("✅ Client initialized successfully")

        # Save uploaded Django files temporarily
        tmp_labeled = None
        tmp_new = None

        if labeled_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(labeled_file.read())
                tmp.flush()
                tmp_labeled = tmp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(new_file.read())
            tmp.flush()
            tmp_new = tmp.name

        # Wrap with handle_file() for Gradio API
        labeled_handle = handle_file(tmp_labeled) if tmp_labeled else None
        new_handle = handle_file(tmp_new)

        for attempt in range(3):
            try:
                print(f"⚙️ Attempt {attempt+1} calling remote ML+AI pipeline...")
                response = client.predict(
                    product_name,
                    description,
                    features,
                    labeled_handle,
                    new_handle,
                    api_name="/predict"   # matches your Hugging Face Space
                )

                if isinstance(response, (list, tuple)):
                    result["ml"] = response[0] if len(response) > 0 else {}
                    result["ai"] = response[1] if len(response) > 1 else {}
                elif isinstance(response, dict):
                    result = response
                else:
                    result["error"] = "Unexpected response format."
                break

            except Exception as e:
                print(f"⚠️ Attempt {attempt+1} failed: {e}")
                time.sleep(2)
        else:
            result["error"] = "Remote inference failed after multiple attempts."

    except Exception as e:
        result["error"] = str(e)
        print("❌ Remote inference error:", e)

    return result
