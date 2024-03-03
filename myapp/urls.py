from django.urls import path
from myapp import views

urlpatterns = [
    path('myapp/', views.myapp, name='myapp'),
]
