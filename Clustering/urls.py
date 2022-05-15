from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.auth.views import LoginView
from django.urls import path, include

from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.data_upload, name="home"),
    path('get-elbow/', views.get_elbow_graph, name="elbow"),
    path('get-clusters/', views.get_clusters, name="clusters")]
