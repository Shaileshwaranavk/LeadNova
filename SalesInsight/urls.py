from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse

def home(request):
    return HttpResponse("<h2>LeadNova API is running successfully ✅</h2>")

urlpatterns = [
    path('', home),  # Root health-check route
    path('admin/', admin.site.urls),
    path('api/', include('core.urls')),  # Your API routes (sales pitch, lead analysis, etc.)
]
