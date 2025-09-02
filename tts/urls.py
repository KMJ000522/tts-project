from django.urls import path
from . import views

urlpatterns = [
    # ex: /tts/
    path('', views.tts_view, name='tts_view'),
    # ex: /tts/health/  (Render.com Health Check용)
    path('health/', views.health_check, name='health_check'),
]