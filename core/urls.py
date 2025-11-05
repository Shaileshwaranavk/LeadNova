from django.urls import path
from .views import (
    SalesPitchAPI,
    LeadAnalysisAPI,
    CRMLeadAnalyzerAPI,
    CRMSalesPitchAPI,
    CRMReviewAPI,
    CRMFeedbackAPI,
    CRMModelRetrainAPI,
    CRMDashboardStatsAPI,
    CRMChartDataAPI,
)

urlpatterns = [
    # === Existing APIs ===
    path("sales-pitch/", SalesPitchAPI.as_view(), name="sales_pitch_api"),
    path("lead-analysis/", LeadAnalysisAPI.as_view(), name="lead_analysis_api"),

    # === CRM APIs ===
    path("crm/analyze-leads/", CRMLeadAnalyzerAPI.as_view(), name="crm_analyze_leads"),
    path("crm/generate-pitch/", CRMSalesPitchAPI.as_view(), name="crm_generate_pitch"),
    path("crm/submit-review/", CRMReviewAPI.as_view(), name="crm_submit_review"),
    path("crm/submit-feedback/", CRMFeedbackAPI.as_view(), name="crm_submit_feedback"),
    path("crm/retrain-models/", CRMModelRetrainAPI.as_view(), name="crm_retrain_models"),
    path("crm/dashboard-stats/", CRMDashboardStatsAPI.as_view(), name="crm_dashboard_stats"),
    path("crm/chart-data/", CRMChartDataAPI.as_view(), name="crm_chart_data"),
]
