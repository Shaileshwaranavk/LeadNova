import os
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.views import View
from django.core.files.storage import default_storage
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from core.infer import generate_recommendation
from core.model_pipeline import run_hybrid_pipeline
import json


@method_decorator(csrf_exempt, name='dispatch')
class SalesPitchAPI(View):
    """
    POST /api/sales-pitch/
    JSON body:
    {
        "product_name": "Smart AI Watch",
        "description": "An intelligent smartwatch for health tracking",
        "features": "AI Assistant, Heart Rate Monitor, Sleep Tracker"
    }
    """
    def post(self, request):
        try:
            data = json.loads(request.body)
            product = data.get("product_name")
            desc = data.get("description")
            features = data.get("features")

            if not all([product, desc, features]):
                return JsonResponse({"error": "Missing required fields"}, status=400)

            result = generate_recommendation(product, desc, features)

            # Clean and structure "feature_weightage"
            feature_lines = [
                line.strip("- ").strip()
                for line in result["feature_weightage"].split("\n")
                if line.strip()
            ]

            structured_features = []
            for line in feature_lines:
                if ":" in line:
                    name, score = line.split(":")
                    structured_features.append({
                        "feature": name.strip(),
                        "score": float(score.strip().replace("/10", "")) if "/10" in score else None
                    })

            # Clean highlighted features into list
            highlighted_list = [
                f.strip() for f in result["highlighted_features"].split(",") if f.strip()
            ]

            # Remove extra quotes/newlines from pitch
            clean_pitch = (
                result["sales_pitch"]
                .replace('\n', ' ')
                .replace('\r', ' ')
                .strip('" ')
            )

            # Final clean JSON response
            return JsonResponse({
                "product": product,
                "feature_weightage": structured_features,
                "highlighted_features": highlighted_list,
                "llm_insight": result["llm_feature_insight"].strip(),
                "sales_pitch": clean_pitch
            }, json_dumps_params={"indent": 2})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class LeadAnalysisAPI(View):
    """
    POST /api/lead-analysis/
    Multipart Form-Data fields:
      - product_name
      - description
      - features
      - labeled_data (file, CSV)
      - new_data (file, CSV)
    """
    def post(self, request):
        try:
            product = request.POST.get("product_name")
            desc = request.POST.get("description")
            features = request.POST.get("features")

            if not all([product, desc, features]):
                return JsonResponse({"error": "Missing text fields"}, status=400)

            labeled_file = request.FILES.get("labeled_data")
            new_file = request.FILES.get("new_data")
            if not labeled_file or not new_file:
                return JsonResponse({"error": "Both CSV files required"}, status=400)

            labeled_path = default_storage.save(labeled_file.name, labeled_file)
            new_path = default_storage.save(new_file.name, new_file)

            labeled_abs = os.path.join(settings.MEDIA_ROOT, labeled_path)
            new_abs = os.path.join(settings.MEDIA_ROOT, new_path)

            result = run_hybrid_pipeline(
                new_data_path=new_abs,
                labeled_data_path=labeled_abs,
                product_name=product,
                description=desc,
                features=features
            )

            return JsonResponse(result, safe=False)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
