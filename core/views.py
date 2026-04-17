from django.shortcuts import render, redirect, get_object_or_404
from .forms import UploadFileForm, AddRecordForm
from .models import Upload, DemandRecord
from .services import read_and_validate
from .models import ForecastRun, ForecastResult
from .services import train_and_forecast, moving_average_forecast, exponential_smoothing_forecast
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
    form = AddRecordForm()
    return render(request, "core/upload_detail.html", {"upload": upload, "records": records, "form": form})


def add_record(request, upload_id):
    upload = get_object_or_404(Upload, id=upload_id)
    form = AddRecordForm(request.POST)
    error = None

    if form.is_valid():
        from .services import _to_month_start
        month = _to_month_start(form.cleaned_data["month"])
        if month is None:
            error = "Invalid month — use YYYY-MM format."
        else:
            demand = form.cleaned_data["demand"]
            obj, created = DemandRecord.objects.get_or_create(
                upload=upload, month=month,
                defaults={"demand": demand},
            )
            if not created:
                obj.demand = demand
                obj.save()

    if error:
        records = DemandRecord.objects.filter(upload=upload).order_by("month")
        return render(request, "core/upload_detail.html", {
            "upload": upload, "records": records, "form": form, "error": error
        })

    return redirect("upload_detail", upload_id=upload.id)


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

    # Moving Average
    ma_out = moving_average_forecast(dates, values, horizon_months=horizon, window=n_lags, test_size=test_size)
    ma_run = ForecastRun.objects.create(
        upload=upload, model_name="Moving Average",
        horizon_months=horizon, n_lags=n_lags, test_size=test_size,
        mae=ma_out["mae"], rmse=ma_out["rmse"], mape=ma_out["mape"],
    )
    ForecastResult.objects.bulk_create([
        ForecastResult(run=ma_run, forecast_month=d, y_pred=p) for d, p in ma_out["preds"]
    ])
    TestResult.objects.bulk_create([
        TestResult(run=ma_run, month=d, y_true=float(y_t), y_pred=float(y_p))
        for d, y_t, y_p in ma_out["test_points"]
    ])

    # Exponential Smoothing
    es_out = exponential_smoothing_forecast(dates, values, horizon_months=horizon, test_size=test_size)
    es_run = ForecastRun.objects.create(
        upload=upload, model_name="Exp. Smoothing",
        horizon_months=horizon, n_lags=0, test_size=test_size,
        mae=es_out["mae"], rmse=es_out["rmse"], mape=es_out["mape"],
    )
    ForecastResult.objects.bulk_create([
        ForecastResult(run=es_run, forecast_month=d, y_pred=p) for d, p in es_out["preds"]
    ])
    TestResult.objects.bulk_create([
        TestResult(run=es_run, month=d, y_true=float(y_t), y_pred=float(y_p))
        for d, y_t, y_p in es_out["test_points"]
    ])

    return redirect("forecast_compare", upload_id=upload.id)


def forecast_compare(request, upload_id):
    upload = get_object_or_404(Upload, id=upload_id)

    # Get latest run per model
    all_model_names = ["Random Forest", "XGBoost", "SVR", "Moving Average", "Exp. Smoothing"]
    latest = {}
    for run in ForecastRun.objects.filter(upload=upload).order_by("-created_at"):
        if run.model_name not in latest:
            latest[run.model_name] = run
        if len(latest) == len(all_model_names):
            break

    run_results = []
    for name in all_model_names:
        run = latest.get(name)
        if run:
            forecast_results = ForecastResult.objects.filter(run=run).order_by("forecast_month")
            test_results = TestResult.objects.filter(run=run).order_by("month")
            run_results.append((run, forecast_results, test_results))
            
    return render(request, "core/forecast_compare.html", {
        "upload": upload,
        "run_results": run_results
    })
