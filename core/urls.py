from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("upload/", views.upload_data, name="upload_data"),
    path("upload/<int:upload_id>/", views.upload_detail, name="upload_detail"),
    path("upload/<int:upload_id>/add-record/", views.add_record, name="add_record"),
    path("upload/<int:upload_id>/run-forecast/", views.run_forecast, name="run_forecast"),
    path("upload/<int:upload_id>/forecast-compare/", views.forecast_compare, name="forecast_compare"),

]
