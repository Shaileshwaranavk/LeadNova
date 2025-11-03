from model_pipeline import run_hybrid_pipeline
import pandas as pd

# ======================================================
# === RUN HYBRID LEAD SCORING PIPELINE ================
# ======================================================
result = run_hybrid_pipeline(
    new_data_path="uploaded_leads.csv",
    labeled_data_path="lead_dataset_with_conversion.csv",
    product_name="Smart Home Hub",
    description="An intelligent IoT device connecting and automating all home appliances.",
    features="AI control, mobile app, voice assistant, energy optimization"
)

# ======================================================
# === FORMATTED & ALIGNED OUTPUT =======================
# ======================================================
print("\n" + "="*70)
print("🚀 HYBRID LEAD SCORING PIPELINE RESULTS")
print("="*70)

# --- Machine Learning Section ---
ml = result.get("ml", {})
print("\n📘 MACHINE LEARNING RESULT")
print("-" * 50)
print(f"Status          : {ml.get('status', 'N/A')}")
print(f"R² Score        : {ml.get('r2_score', 'N/A')}")

if ml.get("top_leads"):
    print("\nTop 5 ML-Predicted Leads:")
    df_ml_top = pd.DataFrame(ml["top_leads"]).head(20)
    print(df_ml_top.to_string(index=False))
else:
    print("No ML leads generated.")

if ml.get("lime_explanation"):
    print("\nKey Feature Influences (LIME Explainability):")
    df_lime = pd.DataFrame(ml["lime_explanation"]).head(20)
    print(df_lime.to_string(index=False))
else:
    print("No LIME explanation available.")

# --- AI Section ---
ai = result.get("ai", {})
print("\n🤖 AI RESULT")
print("-" * 50)
print(f"Status          : {ai.get('status', 'N/A')}")
print(f"Summary         : {ai.get('summary', 'N/A')}")

if ai.get("top_leads"):
    print("\nTop 5 AI-Predicted Leads:")
    df_ai_top = pd.DataFrame(ai["top_leads"]).head(20)
    print(df_ai_top.to_string(index=False))
else:
    print("No AI leads generated.")

# --- Hybrid Section ---
print("\n🔀 HYBRID (Combined) ANALYSIS")
print("-" * 50)
hybrid = result.get("hybrid", [])
if hybrid:
    df_hybrid = pd.DataFrame(hybrid).head(10)
    print(df_hybrid.to_string(index=False))
else:
    print("No hybrid analysis yet implemented.")

print("\n✅ Report generation complete!")
print("="*70)
