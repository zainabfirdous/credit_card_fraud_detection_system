from django.urls import path
#now import the views.py file into this code
from . import views
from django.contrib import admin
from django.contrib.auth import views as auth_views



urlpatterns=[
    path('admin/', admin.site.urls),
    path('',views.login_view),
    path('home/', views.home, name='home'),
    path('home/predict/',views.predict, name='predict'),
    path('home/predict/form', views.single, name='singleentry'),
    path('home/predict/fileupload', views.file, name='fileupload'),
   # path('home/predict/fileupload/analyze', views.file, name='analyze'),
    path('home/predict/result/',views.result, name='reuslt'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
    path('home/about/',views.about, name='about'),
    path('home/contact/',views.contact, name='contact'),
  
]
