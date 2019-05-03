"""web URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.urls import path
from django.contrib import admin
from nbad3 import views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url('coach/', views.coach, name='coach'),
    url('ajax/load_events/', views.load_events, name='ajax_load_events'),
    path('play_anim_data/<str:event_id>', views.play_anim_data, name='play_anim_data'),
    # path('play_anim_data/', views.play_anim_data, name='play_anim_data')
]