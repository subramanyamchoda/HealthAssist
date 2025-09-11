
from django.contrib import admin
from django.urls import path,include
from django.http import JsonResponse


def home(request):
    return JsonResponse({"status": "ok", "message": "Backend is running"})

    
urlpatterns = [
    path('admin/', admin.site.urls),
      path('', home), 
    path('user/',include('health.urls')),
    
]


