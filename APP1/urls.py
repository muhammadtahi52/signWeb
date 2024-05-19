from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('index.html', views.home, name='home'),
    path('service.html', views.service, name='service'),
    path('about.html', views.about, name='about'),
    path('contact.html', views.contact, name='contact'),
    path('signToVoice.html', views.signToVoice, name='signToVoice'),
    path('TextToSign.html', views.TextToSign, name='TextToSign'),
    path('VoiceToSign.html', views.VoiceToSign, name='VoiceToSign'),
    path('signToText.html', views.signToText, name='signToText'),
    path('signToText.html', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('fetch_predicted_sentence/', views.fetch_predicted_sentence, name='fetch_predicted_sentence'),
    path('add_space/', views.add_space, name='add_space'),
    path('delete_letter/', views.delete_letter, name='delete_letter'),

]
