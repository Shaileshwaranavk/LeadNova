from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Lazy global cache
_model = None
_tokenizer = None

def load_sales_model(model_path="./core/sales_model"):
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print("🔄 Loading T5 model into memory (once)...")
        _tokenizer = T5Tokenizer.from_pretrained(model_path)
        _model = T5ForConditionalGeneration.from_pretrained(model_path)
        _model = _model.to("cpu")  # ensure not trying to use GPU
    return _model, _tokenizer


def generate_recommendation(product_name, description, features):
    model, tokenizer = load_sales_model()

    analysis = analyze_features(product_name, description, features)

    prompt = f"""
    Product: {product_name}
    Description: {description}
    Features: {features}
    Generate a brief insight about which features appeal most to target customers.
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128)
    llm_insight = tokenizer.decode(outputs[0], skip_special_tokens=True)

    pitch = generate_sales_pitch(product_name, description, analysis["highlighted"])

    formatted_weightage = analysis["weightage"].replace("\n", " | ")

    return {
        "product": product_name,
        "feature_weightage": formatted_weightage,
        "highlighted_features": analysis["highlighted"],
        "llm_insight": llm_insight,
        "sales_pitch": pitch,
    }
