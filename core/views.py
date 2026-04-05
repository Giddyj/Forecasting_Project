from django.shortcuts import render, redirect, get_object_or_404
from .forms import UploadFileForm
from .models import Upload, DemandRecord
from .services import read_and_validate
from .models import ForecastRun, ForecastResult
from .services import train_and_forecast
from .models import ForecastRun, ForecastResult, TestResult


def home(request):
    uploads = Upload.objects.order_by("-created_at")[:20]
    return render(request, "core/home.html", {"uploads": uploads})

def upload_data(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            f = form.cleaned_data["file"]
            upload = Upload.objects.create(original_filename=f.name, status="imported")

            try:
                df = read_and_validate(f)
                DemandRecord.objects.bulk_create([
                    DemandRecord(upload=upload, month=row["month"], demand=float(row["demand"]))
                    for _, row in df.iterrows()
                ])
            except Exception as e:
                upload.status = "failed"
                upload.note = str(e)
                upload.save()
                return render(request, "core/upload.html", {"form": form, "error": str(e)})

            return redirect("upload_detail", upload_id=upload.id)
    else:
        form = UploadFileForm()

    return render(request, "core/upload.html", {"form": form})

def upload_detail(request, upload_id):
    upload = get_object_or_404(Upload, id=upload_id)
    records = DemandRecord.objects.filter(upload=upload).order_by("month")
    return render(request, "core/upload_detail.html", {"upload": upload, "records": records})


def run_forecast(request, upload_id):
    upload = get_object_or_404(Upload, id=upload_id)

    records = DemandRecord.objects.filter(upload=upload).order_by("month")
    dates = [r.month for r in records]
    values = [float(r.demand) for r in records]

    horizon = 6
    n_lags = 3
    test_size = 6


    for model_name in ["Random Forest", "XGBoost", "SVR"]:
        out = train_and_forecast(
            model_name=model_name,
            dates=dates,
            values=values,
            horizon_months=horizon,
            n_lags=n_lags,
            test_size=test_size,
        )

        run = ForecastRun.objects.create(
            upload=upload,
            model_name=model_name,
            horizon_months=horizon,
            n_lags=n_lags,
            test_size=test_size,
            mae=out["mae"],
            rmse=out["rmse"],
            mape=out["mape"],
        )

        ForecastResult.objects.bulk_create([
            ForecastResult(run=run, forecast_month=d, y_pred=p)
            for d, p in out["preds"]
        ])
        TestResult.objects.bulk_create([
        TestResult(run=run, month=d, y_true=float(y_t), y_pred=float(y_p))
        for d, y_t, y_p in out["test_points"]
        ])

    return redirect("forecast_compare", upload_id=upload.id)


def forecast_compare(request, upload_id):
    upload = get_object_or_404(Upload, id=upload_id)

    # Get latest run per model
    latest = {}
    for run in ForecastRun.objects.filter(upload=upload).order_by("-created_at"):
        if run.model_name not in latest:
            latest[run.model_name] = run
        if len(latest) == 3:
            break

    ordered_models = ["Random Forest", "XGBoost", "SVR"]
    run_results = []
    for name in ["Random Forest", "XGBoost", "SVR"]:
        run = latest.get(name)
        if run:
            forecast_results = ForecastResult.objects.filter(run=run).order_by("forecast_month")
            test_results = TestResult.objects.filter(run=run).order_by("month")
            run_results.append((run, forecast_results, test_results))
            
    return render(request, "core/forecast_compare.html", {
        "upload": upload,
        "run_results": run_results
    })
