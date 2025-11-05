from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.forms.models import model_to_dict
from django.db.models import Count, Avg, Sum
from django.contrib.auth import authenticate
from datetime import datetime, timedelta
import traceback

from rest_framework_simplejwt.tokens import RefreshToken

from core.infer import generate_remote_recommendation
from core.model_pipeline import analyze_leads_remote
from .models import CompanyUser, Customer, CustomerReview, ModelFeedback
from .CRM_Trainer import retrain_from_feedback


# ============================================================
# 🧩 AUTHENTICATION VIEWS (Company Register/Login)
# ============================================================

class RegisterAPI(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            email = request.data.get("email")
            company_name = request.data.get("company_name")
            password = request.data.get("password")

            if not (email and password and company_name):
                return Response({"error": "Email, company_name and password are required."}, status=400)

            if CompanyUser.objects.filter(email=email).exists():
                return Response({"error": "Email already registered."}, status=400)

            user = CompanyUser.objects.create_user(email=email, company_name=company_name, password=password)
            refresh = RefreshToken.for_user(user)
            return Response({
                "message": "Company registered successfully.",
                "company": user.company_name,
                "access": str(refresh.access_token),
                "refresh": str(refresh)
            }, status=201)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)


class LoginAPI(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            email = request.data.get("email")
            password = request.data.get("password")

            user = authenticate(email=email, password=password)
            if user is None:
                return Response({"error": "Invalid email or password."}, status=401)

            refresh = RefreshToken.for_user(user)
            return Response({
                "message": "Login successful.",
                "company": user.company_name,
                "access": str(refresh.access_token),
                "refresh": str(refresh)
            }, status=200)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)


# ============================================================
# 🧩 SALES PITCH & LEAD ANALYZER (Remote Hugging Face)
# ============================================================

class SalesPitchAPI(APIView):
    permission_classes = [IsAuthenticated]

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


class LeadAnalysisAPI(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            product_name = request.data.get("product_name", "").strip()
            description = request.data.get("description", "").strip()
            features = request.data.get("features", "").strip()

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


# ============================================================
# 🧩 CRM REVIEWS, FEEDBACK, AND RETRAINING
# ============================================================

class CRMReviewAPI(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            data = request.data
            customer_id = data.get("customer_id")
            text = data.get("review_text")

            if not (customer_id and text):
                return Response({"error": "Missing customer_id or review_text"}, status=400)

            customer = Customer.objects.get(customer_id=customer_id, owner=request.user)
            review = CustomerReview.objects.create(customer=customer, review_text=text, owner=request.user)
            return Response({"status": "Review saved", "id": review.id}, status=201)
        except Customer.DoesNotExist:
            return Response({"error": "Customer not found or not yours"}, status=403)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)


class CRMFeedbackAPI(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            data = request.data
            fb = ModelFeedback.objects.create(
                owner=request.user,
                model_name=data.get("model_name"),
                feedback_text=data.get("feedback_text"),
                rating=data.get("rating", 0)
            )
            return Response({"status": "Feedback saved", "id": fb.id}, status=201)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)


class CRMModelRetrainAPI(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            result = retrain_from_feedback()
            return Response(result, status=200)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)


# ============================================================
# 🧩 CRM DASHBOARD & CHARTS (Scoped per Company)
# ============================================================

class CRMDashboardStatsAPI(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            total_customers = Customer.objects.filter(owner=request.user).count()
            total_reviews = CustomerReview.objects.filter(owner=request.user).count()
            avg_conversion = (
                Customer.objects.filter(owner=request.user).aggregate(avg_rate=Avg("conversion_rate"))["avg_rate"] or 0
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


class CRMChartDataAPI(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            customers = Customer.objects.filter(owner=request.user)

            # Revenue distribution
            revenue_ranges = [
                {'range': '0-10k', 'count': customers.filter(revenue_potential__lt=10000).count()},
                {'range': '10k-50k', 'count': customers.filter(revenue_potential__gte=10000, revenue_potential__lt=50000).count()},
                {'range': '50k-100k', 'count': customers.filter(revenue_potential__gte=50000, revenue_potential__lt=100000).count()},
                {'range': '100k+', 'count': customers.filter(revenue_potential__gte=100000).count()},
            ]

            top_industries = customers.values('industry').annotate(
                avg_conversion=Avg('conversion_rate'),
                count=Count('customer_id')
            ).filter(industry__isnull=False).order_by('-avg_conversion')[:5]

            return Response({
                'top_industries': list(top_industries),
                'revenue_distribution': revenue_ranges,
                'total_revenue_potential': customers.aggregate(total=Sum('revenue_potential'))['total'] or 0
            }, status=200)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)


# ============================================================
# 🧩 CUSTOMER MANAGEMENT CRUD (Per Company)
# ============================================================

class CustomerListAPI(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            customers = Customer.objects.filter(owner=request.user).values()
            return Response(list(customers), status=200)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)

    def post(self, request):
        try:
            data = request.data
            name = data.get("name")
            email = data.get("email")

            if not name or not email:
                return Response({"error": "Name and email are required."}, status=400)

            if Customer.objects.filter(owner=request.user, email=email).exists():
                return Response({"error": "Customer with this email already exists."}, status=400)

            customer = Customer.objects.create(
                owner=request.user,
                name=name,
                email=email,
                company=data.get("company", ""),
                industry=data.get("industry", ""),
                country=data.get("country", ""),
                product_interested=data.get("product_interested", ""),
                revenue_potential=data.get("revenue_potential", 0),
                conversion_rate=data.get("conversion_rate", 0),
            )
            return Response({"status": "Customer added", "customer": model_to_dict(customer)}, status=201)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)


class CustomerDetailAPI(APIView):
    permission_classes = [IsAuthenticated]

    def get_object(self, request, pk):
        try:
            return Customer.objects.get(pk=pk, owner=request.user)
        except Customer.DoesNotExist:
            return None

    def get(self, request, customer_id):
        customer = self.get_object(request, customer_id)
        if not customer:
            return Response({"error": "Customer not found or not yours"}, status=404)
        return Response(model_to_dict(customer), status=200)

    def put(self, request, customer_id):
        try:
            customer = self.get_object(request, customer_id)
            if not customer:
                return Response({"error": "Customer not found or not yours"}, status=404)

            for field in ["name", "email", "company", "industry", "country",
                          "product_interested", "revenue_potential", "conversion_rate"]:
                if field in request.data:
                    setattr(customer, field, request.data[field])

            customer.save()
            return Response({"status": "Customer updated", "customer": model_to_dict(customer)}, status=200)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)

    def delete(self, request, customer_id):
        try:
            customer = self.get_object(request, customer_id)
            if not customer:
                return Response({"error": "Customer not found or not yours"}, status=404)
            customer.delete()
            return Response({"status": "Customer deleted"}, status=200)
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)
