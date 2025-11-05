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

from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Customer, CustomerReview, ModelFeedback
from .CRM_Trainer import analyze_leads_from_db, generate_sales_pitch, retrain_from_feedback
from django.db.models import Avg
import json
import traceback


# 🧩 --- CRM Dashboard APIs ---


class CRMLeadAnalyzerAPI(APIView):
    """Analyze all customers using Hugging Face Lead Analyzer"""

    def get(self, request):
        try:
            result = analyze_leads_from_db()
            return Response(result, status=200)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)


class CRMSalesPitchAPI(APIView):
    """Generate a custom AI Sales Pitch using Hugging Face"""

    def post(self, request):
        try:
            data = request.data
            product = data.get("product_name", "")
            description = data.get("description", "")
            features = data.get("features", "")
            if not product or not description:
                return Response({"error": "Missing product_name or description"}, status=400)
            result = generate_sales_pitch(product, description, features)
            return Response(result, status=200)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)


class CRMReviewAPI(APIView):
    """Add a customer review to the CRM"""

    def post(self, request):
        try:
            data = request.data
            customer_id = data.get("customer_id")
            text = data.get("review_text")

            if not (customer_id and text):
                return Response({"error": "Missing customer_id or review_text"}, status=400)

            customer = Customer.objects.get(customer_id=customer_id)
            review = CustomerReview.objects.create(customer=customer, review_text=text)
            return Response({"status": "Review saved", "id": review.id}, status=201)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)


class CRMFeedbackAPI(APIView):
    """Submit user feedback for model improvement"""

    def post(self, request):
        try:
            data = request.data
            model_name = data.get("model_name")
            feedback_text = data.get("feedback_text")
            rating = data.get("rating", 0)
            fb = ModelFeedback.objects.create(model_name=model_name, feedback_text=feedback_text, rating=rating)
            return Response({"status": "Feedback saved", "id": fb.id}, status=201)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)


class CRMModelRetrainAPI(APIView):
    """Trigger CRM retraining process (stub or background)"""

    def post(self, request):
        try:
            result = retrain_from_feedback()
            return Response(result, status=200)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)


class CRMDashboardStatsAPI(APIView):
    """Return CRM metrics for dashboard overview"""

    def get(self, request):
        try:
            total_customers = Customer.objects.count()
            total_reviews = CustomerReview.objects.count()
            avg_conversion = (
                Customer.objects.aggregate(avg_rate=Avg("conversion_rate"))["avg_rate"] or 0
            )

            data = {
                "total_customers": total_customers,
                "total_reviews": total_reviews,
                "average_conversion_rate": round(avg_conversion, 2),
            }
            return Response(data, status=200)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)

from django.db.models import Count, Avg, Sum
from datetime import datetime, timedelta

class CRMChartDataAPI(APIView):
    def get(self, request):
        try:
            from core.models import Customer
            
            # Last 7 days conversion data
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=6)
            
            daily_data = []
            for i in range(7):
                date = start_date + timedelta(days=i)
                day_name = date.strftime('%a')
                
                # Calculate average conversion for all customers
                # (In production, filter by date if you have timestamps)
                avg_conversion = Customer.objects.aggregate(
                    avg_rate=Avg('conversion_rate')
                )['avg_rate'] or 0
                
                daily_data.append({
                    'date': day_name,
                    'conversion_rate': round(avg_conversion * 100, 1)
                })
            
            # Top industries
            top_industries = Customer.objects.values('industry').annotate(
                avg_conversion=Avg('conversion_rate'),
                count=Count('customer_id')
            ).filter(industry__isnull=False).order_by('-avg_conversion')[:5]
            
            # Revenue distribution
            revenue_ranges = [
                {'range': '0-10k', 'count': Customer.objects.filter(revenue_potential__lt=10000).count()},
                {'range': '10k-50k', 'count': Customer.objects.filter(revenue_potential__gte=10000, revenue_potential__lt=50000).count()},
                {'range': '50k-100k', 'count': Customer.objects.filter(revenue_potential__gte=50000, revenue_potential__lt=100000).count()},
                {'range': '100k+', 'count': Customer.objects.filter(revenue_potential__gte=100000).count()},
            ]
            
            return Response({
                'daily_conversions': daily_data,
                'top_industries': list(top_industries),
                'revenue_distribution': revenue_ranges,
                'total_revenue_potential': Customer.objects.aggregate(
                    total=Sum('revenue_potential')
                )['total'] or 0
            }, status=200)
            
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)
