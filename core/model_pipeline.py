import pandas as pd
import numpy as np
import requests
import json
import time
import re
import random
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from io import StringIO


# =========================================================
# === GLOBAL RANDOM SEED (For Deterministic Behavior) =====
# =========================================================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# =========================================================
# === FUNCTION: MACHINE LEARNING TRAINING & PREDICTION ====
# =========================================================
def train_and_predict_with_ml(df_train, df_new, product_name, description, features, top_n=100):
    """
    Train a RandomForest regression model using labeled data and predict conversion scores for new leads.
    Includes LIME explainability for the top lead.
    Deterministic output ensured by fixed random seeds.
    """
    result = {"status": "", "r2_score": None, "top_leads": [], "lime_explanation": []}
    target_col = "Conversion_Rate"

    if target_col not in df_train.columns:
        result["status"] = f"❌ Target column '{target_col}' missing in labeled dataset!"
        return result

    if "Product_Interest_Level" in df_train.columns:
        df_train = df_train.drop(columns=["Product_Interest_Level"])

    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]

    # --- Stable keyword filtering (sorted keywords to avoid order randomness)
    keywords = sorted([kw.lower() for kw in [product_name] + description.split() + features.split(",")])
    keyword_hits = df_train["Industry"].astype(str).apply(
        lambda x: any(kw in x.lower() for kw in keywords if isinstance(x, str))
    )
    filtered_df = df_train[keyword_hits]

    if len(filtered_df) < 10:
        X_train_data = X
        y_train_data = y
    else:
        X_train_data = filtered_df.drop(columns=[target_col])
        y_train_data = filtered_df[target_col]

    # --- Align columns
    missing_cols = [c for c in X_train_data.columns if c not in df_new.columns]
    extra_cols = [c for c in df_new.columns if c not in X_train_data.columns]
    for col in missing_cols:
        df_new[col] = 0
    if extra_cols:
        df_new = df_new.drop(columns=extra_cols)

    # --- Encode categorical features (consistent encoding)
    for col in X_train_data.select_dtypes(include=["object"]).columns:
        if col not in df_new.columns:
            df_new[col] = "Unknown"
        le = LabelEncoder()
        all_values = sorted(set(list(X_train_data[col].astype(str)) + list(df_new[col].astype(str))))
        le.fit(all_values)
        X_train_data[col] = le.transform(X_train_data[col].astype(str))
        df_new[col] = le.transform(df_new[col].astype(str))

    # --- Train RandomForest model (fully deterministic)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_data, y_train_data, test_size=0.2, random_state=SEED, shuffle=True
    )
    model = RandomForestRegressor(n_estimators=200, random_state=SEED, bootstrap=True)
    model.fit(X_train, y_train)

    # --- Evaluate R²
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    result["r2_score"] = round(r2, 3)

    # --- Predict on new data
    predictions = model.predict(df_new[X_train_data.columns])
    df_new["Predicted_Conversion_Score_ML"] = predictions

    # --- Sort & format output
    top_leads = df_new.sort_values(by="Predicted_Conversion_Score_ML", ascending=False).head(top_n)
    cols = ["Lead_ID", "Predicted_Conversion_Score_ML"]
    if "Company_Name" in top_leads.columns:
        cols.insert(1, "Company_Name")
    result["top_leads"] = top_leads[cols].to_dict(orient="records")

    # --- LIME Explainability (deterministic)
    try:
        if len(X_train_data) > 0:
            explainer_lime = LimeTabularExplainer(
                training_data=np.array(X_train_data),
                feature_names=X_train_data.columns.tolist(),
                mode="regression",
                random_state=SEED
            )
            top_lead_instance = np.array(top_leads[X_train_data.columns].iloc[0])
            exp = explainer_lime.explain_instance(top_lead_instance, model.predict, num_features=8)

            explanations = []
            for feature, weight in exp.as_list():
                direction = "increases" if weight > 0 else "decreases"
                explanations.append({
                    "feature": feature,
                    "direction": direction,
                    "weight": round(abs(weight), 3)
                })
            result["lime_explanation"] = explanations
    except Exception as e:
        print(f"LIME explanation error: {e}")
        result["lime_explanation"] = []

    result["status"] = "✅ ML prediction successful"
    return result


# =========================================================
# === FUNCTION: AI-BASED LEAD SCORING VIA GROQ API ========
# =========================================================
def analyze_with_ai(product_name, description, features, df, top_n=100):
    """
    Deterministic AI scoring via Groq API (temperature=0)
    with per-lead explanations and correct Company_Name merging.
    """
    from dotenv import load_dotenv
    import os

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing GROQ_API_KEY in .env or environment variables.")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    BATCH_SIZE = 25
    num_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    all_rows = []

    for i in range(num_batches):
        batch = df.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

        # --- Prepare structured per-lead data ---
        leads_text = "\n".join(
            f"{row.get('Lead_ID','')}: Company={row.get('Company_Name','')}, "
            f"Source={row.get('Lead_Source','')}, Country={row.get('Country','')}, "
            f"Revenue={row.get('Annual_Revenue','')}, Employees={row.get('Employee_Count','')}, "
            f"Website_Visits={row.get('Website_Visits','')}, Email_Opens={row.get('Email_Opens','')}"
            for _, row in batch.iterrows()
        )

        # --- Explicit, per-lead instruction ---
        prompt = f"""
You are an expert AI sales analyst.
Product: {product_name}
Description: {description}
Features: {features}

For EACH lead below, output ONE separate line in CSV with this exact format:
Lead_ID,Predicted_Conversion_Score,Explanation

Rules:
- Give realistic scores between 0 and 100.
- Explanation must be *unique* per lead and based on its details (e.g. country, revenue, visits, opens, etc.).
- Do not repeat the same explanation across leads.
- Do NOT include code blocks, headers, or notes — only clean CSV.

Leads:
{leads_text}
"""

        data = {
            "model": "llama-3.1-8b-instant",
            "temperature": 0,
            "top_p": 1,
            "messages": [{"role": "user", "content": prompt}],
        }

        # --- API call with retry ---
        for attempt in range(3):
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60,
                )
                result_json = response.json()
                if "error" in result_json:
                    print(f"API Error: {result_json['error']}")
                    time.sleep(5 * (attempt + 1))
                    continue

                content = result_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not content.strip():
                    time.sleep(2)
                    continue

                lines = [
                    l.strip()
                    for l in content.split("\n")
                    if l.strip() and not l.startswith("```") and "," in l
                ]

                for line in lines:
                    parts = [x.strip() for x in re.split(r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", line)]
                    if len(parts) >= 3:
                        lead_id = str(parts[0])
                        score_match = re.search(r"(\d+\.?\d*)", parts[1])
                        score = float(score_match.group(1)) if score_match else 0.0
                        explanation = parts[2].strip().strip('"')

                        all_rows.append({
                            "Lead_ID": lead_id,
                            "Predicted_Conversion_Score_AI": score,
                            "Explanation": explanation,
                        })
                break  # success, move to next batch
            except Exception as e:
                print(f"Batch {i + 1} retry {attempt + 1} failed: {e}")
                time.sleep(3)

    # --- Convert to DataFrame ---
    df_ai = pd.DataFrame(all_rows)

    # --- Merge Company_Name safely ---
    if "Company_Name" in df.columns:
        df_ai = pd.merge(df_ai, df[["Lead_ID", "Company_Name"]], on="Lead_ID", how="left")

    if df_ai.empty:
        return {
            "status": "⚠ AI scoring - No valid responses",
            "top_leads": [],
            "summary": "No leads processed by AI."
        }

    # --- Clean and deduplicate ---
    df_ai = df_ai.drop_duplicates(subset=["Lead_ID"])
    df_ai["Explanation"] = df_ai["Explanation"].replace("", "No specific reason provided.")

    # --- Sort and select ---
    df_top = df_ai.sort_values(by="Predicted_Conversion_Score_AI", ascending=False).head(top_n)

    cols = ["Lead_ID", "Predicted_Conversion_Score_AI", "Explanation"]
    if "Company_Name" in df_ai.columns:
        cols.insert(1, "Company_Name")

    return {
        "status": "✅ AI scoring complete",
        "top_leads": df_top[cols].to_dict(orient="records"),
        "summary": f"Analyzed {len(df_ai)} valid leads using AI."
    }



# =========================================================
# === MAIN HYBRID PIPELINE ================================
# =========================================================
def run_hybrid_pipeline(new_data_path, labeled_data_path, product_name, description, features, top_n=100):
    try:
        df_new = pd.read_csv(new_data_path)
    except Exception as e:
        return {"ml": None, "ai": None, "hybrid": [], "error": f"Failed to read new_data CSV: {e}"}

    # --- Run ML
    df_ml_result = None
    if labeled_data_path:
        try:
            df_train = pd.read_csv(labeled_data_path)
            df_ml_result = train_and_predict_with_ml(
                df_train, df_new.copy(), product_name, description, features, top_n
            )
        except Exception as e:
            print(f"ML training error: {e}")
            df_ml_result = {"status": f"❌ ML error: {e}", "top_leads": [], "lime_explanation": []}

    # --- Run AI
    try:
        df_ai_result = analyze_with_ai(product_name, description, features, df_new, top_n)
    except Exception as e:
        print(f"AI analysis error: {e}")
        df_ai_result = {"status": f"⚠ AI error: {e}", "top_leads": []}

    # --- Combine Hybrid Scores
    result = {"ml": df_ml_result, "ai": df_ai_result, "hybrid": []}
    try:
        if df_ml_result and df_ai_result and df_ml_result.get("top_leads") and df_ai_result.get("top_leads"):
            df_ml_top = pd.DataFrame(df_ml_result["top_leads"])
            df_ai_top = pd.DataFrame(df_ai_result["top_leads"])
            df_ml_top["Lead_ID"] = df_ml_top["Lead_ID"].astype(str)
            df_ai_top["Lead_ID"] = df_ai_top["Lead_ID"].astype(str)
            merged = pd.merge(df_ai_top, df_ml_top, on="Lead_ID", how="inner")

            if not merged.empty:
                merged["Hybrid_Score"] = (
                    0.6 * merged["Predicted_Conversion_Score_ML"] +
                    0.4 * merged["Predicted_Conversion_Score_AI"]
                )
                merged = merged.sort_values(by="Hybrid_Score", ascending=False)
                result["hybrid"] = merged.to_dict(orient="records")
    except Exception as e:
        print(f"Hybrid merge error: {e}")

    return result
