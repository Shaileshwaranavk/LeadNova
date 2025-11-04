from rest_framework.views import APIView
from rest_framework.response import Response
from core.infer import generate_remote_recommendation
from core.model_pipeline import analyze_leads_remote
import traceback


class SalesPitchAPI(APIView):
    def post(self, request):
        try:
            data = request.data
            product_name = data.get("product_name", "").strip()
            description = data.get("description", "").strip()
            features = data.get("features", "").strip()

            if not product_name or not description:
                return Response({"error": "Missing product_name or description"}, status=400)

            result = generate_remote_recommendation(product_name, description, features)
            return Response(result, status=200)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)

from rest_framework.views import APIView
from rest_framework.response import Response
from core.model_pipeline import analyze_leads_remote
import traceback

class LeadAnalysisAPI(APIView):
    def post(self, request):
        try:
            # 🔍 Debug info
            print("🧩 FILES RECEIVED:", request.FILES.keys())
            print("📦 DATA RECEIVED:", request.data)

            product_name = request.data.get("product_name", "").strip()
            description = request.data.get("description", "").strip()
            features = request.data.get("features", "").strip()

            # 🔑 Accept both key name styles
            labeled_file = (
                request.FILES.get("labeled_csv")
                or request.FILES.get("labeled_data")
                or request.FILES.get("labeled")
            )
            new_file = (
                request.FILES.get("new_csv")
                or request.FILES.get("new_data")
                or request.FILES.get("new")
            )

            if not new_file:
                return Response({"error": "new_csv file is required"}, status=400)

            result = analyze_leads_remote(
                product_name=product_name,
                description=description,
                features=features,
                labeled_file=labeled_file,
                new_file=new_file
            )

            return Response(result, status=200)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)
