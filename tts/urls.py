# tts/urls.py (새로 만드는 파일)

from django.urls import path
from . import views

urlpatterns = [
    # 'tts/generate/' 주소로 요청이 오면 views.py의 generate_speech 함수를 실행
    path('generate/', views.generate_speech, name='generate_speech'),
]