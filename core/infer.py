import os
import requests
import numpy as np

# ===== 1️⃣ Load Groq API Key =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ===== 2️⃣ Feature Analysis =====
def analyze_features(product_name, description, features):
    feature_list = [f.strip() for f in features.split(",") if f.strip()]
    if not feature_list:
        return {"weightage": "N/A", "highlighted": "N/A"}

    desc_words = description.lower().split()
    weights = []
    for f in feature_list:
        score = sum(1 for w in f.lower().split() if w in desc_words)
        weights.append(score or np.random.randint(4, 9))

    max_score = max(weights) if weights else 1
    weights = [round((w / max_score) * 10, 1) for w in weights]
    scored_features = sorted(zip(feature_list, weights), key=lambda x: x[1], reverse=True)
    top_features = [f for f, _ in scored_features[:3]]

    weightage_text = "\n".join([f"- {f}: {s}/10" for f, s in scored_features])
    highlighted_text = ", ".join(top_features)
    return {"weightage": weightage_text, "highlighted": highlighted_text}


# ===== 3️⃣ Generate Pitch via Groq =====
def generate_sales_pitch(product_name, description, highlighted_features):
    if not GROQ_API_KEY:
        return "⚠️ Missing GROQ_API_KEY environment variable."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = f"""
    You are a persuasive sales strategist.
    Write a 3–4 sentence compelling pitch for the product below.
    Product: {product_name}
    Description: {description}
    Highlighted Features: {highlighted_features}
    Tone: confident, clear, and customer-centric.
    """

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload
        )
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "⚠️ No pitch generated.")
    except Exception as e:
        return f"⚠️ API Error: {e}"


# ===== 4️⃣ Recommendation Pipeline (Lazy model load) =====
def generate_recommendation(product_name, description, features):
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch

    # Lazy load: only when function runs
    model_path = "./core/sales_model"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    analysis = analyze_features(product_name, description, features)

    prompt = f"""
    Product: {product_name}
    Description: {description}
    Features: {features}
    Generate a brief insight about which features appeal most to target customers.
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=128)
    llm_insight = tokenizer.decode(outputs[0], skip_special_tokens=True)

    pitch = generate_sales_pitch(product_name, description, analysis["highlighted"])

    # Clean feature text for JSON
    formatted_weightage = analysis["weightage"].replace("\n", " | ")

    return {
        "product": product_name,
        "feature_weightage": formatted_weightage,
        "highlighted_features": analysis["highlighted"],
        "llm_insight": llm_insight,
        "sales_pitch": pitch,
    }


if __name__ == "__main__":
    print("\n💡 Interactive Sales Pitch Generator\n")
    product = input("Product Name: ")
    desc = input("Product Description: ")
    feats = input("Features (comma-separated): ")
    res = generate_recommendation(product, desc, feats)
    print("\n✅ Result:\n", res)
