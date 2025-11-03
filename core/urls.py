from django.urls import path
from . import views

urlpatterns = [
    path('sales-pitch/', views.SalesPitchAPI.as_view(), name='sales_pitch_api'),
    path('lead-analysis/', views.LeadAnalysisAPI.as_view(), name='lead_analysis_api'),
]
