from django.db import models

class Upload(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    original_filename = models.CharField(max_length=255)
    status = models.CharField(max_length=50, default="imported")  # imported / failed
    note = models.TextField(blank=True, default="")

    def __str__(self):
        return f"{self.original_filename} ({self.created_at:%Y-%m-%d %H:%M})"

class DemandRecord(models.Model):
    upload = models.ForeignKey(Upload, on_delete=models.CASCADE, related_name="records")
    month = models.DateField()  # store as first day of the month
    demand = models.FloatField()

    class Meta:
        unique_together = ("upload", "month")
        ordering = ["month"]

    def __str__(self):
        return f"{self.month} = {self.demand}"

class ForecastRun(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    upload = models.ForeignKey(Upload, on_delete=models.CASCADE, related_name="forecast_runs")
    model_name = models.CharField(max_length=50)  # "Random Forest", "XGBoost", "SVR"
    horizon_months = models.IntegerField(default=6)

    # metrics (computed on a holdout set)
    mae = models.FloatField(null=True, blank=True)
    rmse = models.FloatField(null=True, blank=True)
    mape = models.FloatField(null=True, blank=True)

    # cross-validation metrics (TimeSeriesSplit on training data, ML models only)
    cv_mae = models.FloatField(null=True, blank=True)
    cv_rmse = models.FloatField(null=True, blank=True)
    cv_mape = models.FloatField(null=True, blank=True)

    # settings used for evaluation/forecasting
    test_size = models.IntegerField(default=6)
    n_lags = models.IntegerField(default=3)

    def __str__(self):
        return f"{self.model_name} ({self.created_at:%Y-%m-%d %H:%M})"


class ForecastResult(models.Model):
    run = models.ForeignKey(ForecastRun, on_delete=models.CASCADE, related_name="results")
    forecast_month = models.DateField()
    y_pred = models.FloatField()

    class Meta:
        unique_together = ("run", "forecast_month")
        ordering = ["forecast_month"]

class TestResult(models.Model):
    run = models.ForeignKey(ForecastRun, on_delete=models.CASCADE, related_name="test_results")
    month = models.DateField()
    y_true = models.FloatField()
    y_pred = models.FloatField()

    class Meta:
        unique_together = ("run", "month")
        ordering = ["month"]
