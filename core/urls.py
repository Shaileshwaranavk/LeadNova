from django.urls import path
from .views import (
    RegisterAPI,
    LoginAPI,
    SalesPitchAPI,
    LeadAnalysisAPI,
    CRMReviewAPI,
    CRMFeedbackAPI,
    CRMModelRetrainAPI,
    CRMDashboardStatsAPI,
    CRMChartDataAPI,
    CustomerListAPI,
    CustomerDetailAPI,
)

urlpatterns = [
    path("auth/register/", RegisterAPI.as_view(), name="register"),
    path("auth/login/", LoginAPI.as_view(), name="login"),
    path("sales-pitch/", SalesPitchAPI.as_view(), name="sales_pitch"),
    path("lead-analysis/", LeadAnalysisAPI.as_view(), name="lead_analysis"),
    path("crm/review/", CRMReviewAPI.as_view(), name="crm_review"),
    path("crm/feedback/", CRMFeedbackAPI.as_view(), name="crm_feedback"),
    path("crm/retrain/", CRMModelRetrainAPI.as_view(), name="crm_retrain"),
    path("crm/dashboard-stats/", CRMDashboardStatsAPI.as_view(), name="crm_dashboard_stats"),
    path("crm/chart-data/", CRMChartDataAPI.as_view(), name="crm_chart_data"),
    path("crm/customers/", CustomerListAPI.as_view(), name="crm_customers"),
    path("crm/customers/<int:customer_id>/", CustomerDetailAPI.as_view(), name="crm_customer_detail"),
]
