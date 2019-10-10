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
from django.urls import path, re_path
from django.contrib import admin
from nbad3 import views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url('coach/', views.coach, name='coach'),
    url('ajax/load_possessions/', views.load_possessions, name='ajax_load_possessions'),
    re_path(r'^play_anim_data/(?P<possession_id>[0-9]+)?/(?P<half_court>.*)', views.play_anim_data, name='play_anim_data'),
    # path('play_anim_data/', views.play_anim_data, name='play_anim_data')
]
