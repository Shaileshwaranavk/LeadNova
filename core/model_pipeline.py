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
    Trains a RandomForest model each time with deterministic behavior.
    Predicts scores for df_new, explains via LIME.
    """
    result = {"status": "", "r2_score": None, "top_leads": [], "lime_explanation": []}
    target_col = "Conversion_Rate"

    if target_col not in df_train.columns:
        result["status"] = f"❌ Target column '{target_col}' missing!"
        return result

    if "Product_Interest_Level" in df_train.columns:
        df_train = df_train.drop(columns=["Product_Interest_Level"])

    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]

    # --- Deterministic filtering
    keywords = sorted([kw.lower() for kw in [product_name] + description.split() + features.split(",")])
    mask = df_train["Industry"].astype(str).apply(
        lambda x: any(kw in x.lower() for kw in keywords if isinstance(x, str))
    )
    filtered = df_train[mask]

    if len(filtered) >= 10:
        X_train_data = filtered.drop(columns=[target_col])
        y_train_data = filtered[target_col]
    else:
        X_train_data, y_train_data = X, y

    # --- Column alignment
    missing_cols = [c for c in X_train_data.columns if c not in df_new.columns]
    extra_cols = [c for c in df_new.columns if c not in X_train_data.columns]
    for c in missing_cols:
        df_new[c] = 0
    if extra_cols:
        df_new = df_new.drop(columns=extra_cols)

    # --- Deterministic encoding
    for col in X_train_data.select_dtypes(include=["object"]).columns:
        if col not in df_new.columns:
            df_new[col] = "Unknown"
        le = LabelEncoder()
        all_vals = sorted(set(list(X_train_data[col].astype(str)) + list(df_new[col].astype(str))))
        le.fit(all_vals)
        X_train_data[col] = le.transform(X_train_data[col].astype(str))
        df_new[col] = le.transform(df_new[col].astype(str))

    # --- Train model (deterministic)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_data, y_train_data, test_size=0.2, random_state=SEED, shuffle=True
    )
    model = RandomForestRegressor(
        n_estimators=150, max_depth=10, random_state=SEED, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # --- Evaluate
    r2 = r2_score(y_test, model.predict(X_test))
    result["r2_score"] = round(r2, 3)

    # --- Predict on new leads
    preds = model.predict(df_new[X_train_data.columns])
    df_new["Predicted_Conversion_Score_ML"] = preds

    # --- Sort & pick top leads
    top = df_new.sort_values(by="Predicted_Conversion_Score_ML", ascending=False).head(top_n)
    cols = ["Lead_ID", "Predicted_Conversion_Score_ML"]
    if "Company_Name" in top.columns:
        cols.insert(1, "Company_Name")
    result["top_leads"] = top[cols].to_dict(orient="records")

    # --- LIME Explanation
    try:
        if len(X_train_data) > 0:
            explainer = LimeTabularExplainer(
                training_data=np.array(X_train_data),
                feature_names=X_train_data.columns.tolist(),
                mode="regression",
                random_state=SEED
            )
            top_lead_instance = np.array(top[X_train_data.columns].iloc[0])
            exp = explainer.explain_instance(top_lead_instance, model.predict, num_features=8)
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
        print(f"LIME error: {e}")
        result["lime_explanation"] = []

    result["status"] = "✅ ML training and prediction complete"
    return result


# =========================================================
# === FUNCTION: AI SCORING USING GROQ =====================
# =========================================================
def analyze_with_ai(product_name, description, features, df, top_n=100):
    from dotenv import load_dotenv
    import os
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"status": "⚠ Missing GROQ_API_KEY", "top_leads": [], "summary": "No key."}

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    all_rows = []
    BATCH_SIZE = 25

    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i + BATCH_SIZE]
        leads_text = "\n".join(
            f"{r.get('Lead_ID','')}: {r.get('Company_Name','')} - {r.get('Industry','')}, "
            f"Revenue={r.get('Annual_Revenue','')}, Visits={r.get('Website_Visits','')}, Opens={r.get('Email_Opens','')}"
            for _, r in batch.iterrows()
        )

        prompt = f"""
You are an AI sales analyst.
Product: {product_name}
Description: {description}
Features: {features}

For each lead, output CSV: Lead_ID,Predicted_Conversion_Score,Explanation
Scores: 0–100 realistic range.
Each explanation must be specific (not generic).
{leads_text}
"""
        data = {
            "model": "llama-3.1-8b-instant",
            "temperature": 0,
            "top_p": 1,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            res = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                headers=headers, json=data, timeout=60)
            res_json = res.json()
            content = res_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                continue

            lines = [l.strip() for l in content.split("\n") if "," in l and not l.startswith("```")]
            for line in lines:
                parts = [p.strip() for p in re.split(r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", line)]
                if len(parts) >= 3:
                    lead_id = parts[0]
                    match = re.search(r"(\d+\.?\d*)", parts[1])
                    score = float(match.group(1)) if match else 0.0
                    explanation = parts[2].strip().strip('"')
                    all_rows.append({
                        "Lead_ID": lead_id,
                        "Predicted_Conversion_Score_AI": score,
                        "Explanation": explanation
                    })
        except Exception as e:
            print(f"AI batch error: {e}")
            time.sleep(3)

    df_ai = pd.DataFrame(all_rows)
    if "Company_Name" in df.columns and not df_ai.empty:
        df_ai = pd.merge(df_ai, df[["Lead_ID", "Company_Name"]], on="Lead_ID", how="left")
    if df_ai.empty:
        return {"status": "⚠ AI scoring failed", "top_leads": []}

    df_ai = df_ai.drop_duplicates(subset=["Lead_ID"])
    df_ai["Explanation"] = df_ai["Explanation"].replace("", "No reason given.")
    df_top = df_ai.sort_values(by="Predicted_Conversion_Score_AI", ascending=False).head(top_n)

    cols = ["Lead_ID", "Predicted_Conversion_Score_AI", "Explanation"]
    if "Company_Name" in df_ai.columns:
        cols.insert(1, "Company_Name")

    return {"status": "✅ AI scoring complete",
            "top_leads": df_top[cols].to_dict(orient="records"),
            "summary": f"Analyzed {len(df_ai)} leads."}


# =========================================================
# === MAIN HYBRID PIPELINE ================================
# =========================================================
def run_hybrid_pipeline(new_data_path, labeled_data_path, product_name, description, features, top_n=100):
    try:
        df_new = pd.read_csv(new_data_path)
    except Exception as e:
        return {"ml": None, "ai": None, "hybrid": [], "error": f"Failed to read new_data: {e}"}

    # --- ML Part
    ml_result = {"status": "⚠ Skipped ML", "top_leads": []}
    try:
        df_train = pd.read_csv(labeled_data_path)
        ml_result = train_and_predict_with_ml(df_train, df_new.copy(), product_name, description, features, top_n)
    except Exception as e:
        ml_result = {"status": f"❌ ML error: {e}", "top_leads": []}

    # --- AI Part
    try:
        ai_result = analyze_with_ai(product_name, description, features, df_new, top_n)
    except Exception as e:
        ai_result = {"status": f"⚠ AI error: {e}", "top_leads": []}

    # --- Combine
    hybrid = []
    try:
        df_ml = pd.DataFrame(ml_result.get("top_leads", []))
        df_ai = pd.DataFrame(ai_result.get("top_leads", []))
        if not df_ml.empty and not df_ai.empty:
            df_ml["Lead_ID"] = df_ml["Lead_ID"].astype(str)
            df_ai["Lead_ID"] = df_ai["Lead_ID"].astype(str)
            merged = pd.merge(df_ml, df_ai, on="Lead_ID", how="inner")
            if not merged.empty:
                merged["Hybrid_Score"] = (
                    0.6 * merged["Predicted_Conversion_Score_ML"] +
                    0.4 * merged["Predicted_Conversion_Score_AI"]
                )
                merged = merged.sort_values(by="Hybrid_Score", ascending=False)
                hybrid = merged.to_dict(orient="records")
    except Exception as e:
        print(f"Hybrid merge error: {e}")

    return {"ml": ml_result, "ai": ai_result, "hybrid": hybrid}
