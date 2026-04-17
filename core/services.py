import pandas as pd

def _to_month_start(x):
    dt = pd.to_datetime(x, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.to_period("M").to_timestamp().date()

def read_and_validate(file_obj):
    name = getattr(file_obj, "name", "").lower()

    if name.endswith(".csv"):
        df = pd.read_csv(file_obj)
    else:
        df = pd.read_excel(file_obj)

    df.columns = [c.strip().lower() for c in df.columns]

    required = {"month", "demand"}
    if not required.issubset(set(df.columns)):
        raise ValueError("File must contain columns: month, demand")

    df = df[["month", "demand"]].copy()

    df["month"] = df["month"].apply(_to_month_start)
    if df["month"].isna().any():
        raise ValueError("Some 'month' values could not be parsed. Use YYYY-MM or a valid date.")

    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    if df["demand"].isna().any():
        raise ValueError("Some 'demand' values are not numeric.")

    df = df.drop_duplicates(subset=["month"], keep="last")
    df = df.sort_values("month").reset_index(drop=True)

    if len(df) < 6:
        raise ValueError("Need at least 6 monthly rows to proceed.")

    return df


import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from xgboost import XGBRegressor


def make_supervised(values, n_lags=3):
    X, y = [], []
    for i in range(n_lags, len(values)):
        X.append(values[i - n_lags:i])
        y.append(values[i])
    return np.array(X, dtype=float), np.array(y, dtype=float)


def compute_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0) if mask.any() else None

    return mae, rmse, mape


def train_and_forecast(model_name, dates, values, horizon_months=6, n_lags=3, test_size=6):
    n = len(values)

    # auto-adjust for small datasets so demo files work
    n_lags = min(n_lags, max(1, n - 3))          # keep at least 1 lag, leave some room
    test_size = min(test_size, max(1, n - n_lags - 1))  # keep at least 1 test point

    # still need enough data to create supervised samples
    if n <= (n_lags + 1):
        raise ValueError("Not enough rows even after adjustment. Upload more months.")

    X, y = make_supervised(values, n_lags=n_lags)

    split_idx = len(y) - test_size
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    if model_name == "SVR":
        model = SVR()
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=300, random_state=42)
    elif model_name == "XGBoost":
        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    else:
        raise ValueError("Unknown model")

    # TimeSeriesSplit CV on the training portion only
    n_splits = min(5, max(2, split_idx // 2))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_maes, cv_rmses, cv_mapes = [], [], []
    for train_idx, val_idx in tscv.split(X_train):
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        model.fit(X_train[train_idx], y_train[train_idx])
        y_val_pred = model.predict(X_train[val_idx])
        f_mae, f_rmse, f_mape = compute_metrics(y_train[val_idx], y_val_pred)
        cv_maes.append(f_mae)
        cv_rmses.append(f_rmse)
        if f_mape is not None:
            cv_mapes.append(f_mape)

    cv_mae = float(np.mean(cv_maes)) if cv_maes else None
    cv_rmse = float(np.mean(cv_rmses)) if cv_rmses else None
    cv_mape = float(np.mean(cv_mapes)) if cv_mapes else None

    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    mae, rmse, mape = compute_metrics(y_test, y_pred_test)

    # Map supervised indices back to actual months:
    # y[i] corresponds to dates[n_lags + i]
    test_start_in_y = split_idx
    test_end_in_y = len(y)
    test_dates = dates[n_lags + test_start_in_y : n_lags + test_end_in_y]

    test_points = list(zip(test_dates, y_test.tolist(), y_pred_test.tolist()))

    # retrain on all data
    model.fit(X, y)

    history = list(map(float, values))
    last_date = dates[-1]
    preds = []

    for _ in range(horizon_months):
        x_input = np.array(history[-n_lags:], dtype=float).reshape(1, -1)
        yhat = float(model.predict(x_input)[0])
        next_month = (pd.Timestamp(last_date) + pd.offsets.MonthBegin(1)).date()
        preds.append((next_month, yhat))
        history.append(yhat)
        last_date = next_month

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "cv_mae": cv_mae,
        "cv_rmse": cv_rmse,
        "cv_mape": cv_mape,
        "preds": preds,
        "test_points": test_points,
    }


def moving_average_forecast(dates, values, horizon_months=6, window=3, test_size=6):
    n = len(values)
    values = list(map(float, values))
    test_size = min(test_size, max(1, n - window - 1))
    train_end = n - test_size

    test_points = []
    for i in range(test_size):
        idx = train_end + i
        w = min(window, idx)
        y_pred = float(np.mean(values[idx - w:idx]))
        test_points.append((dates[idx], values[idx], y_pred))

    y_true = [t[1] for t in test_points]
    y_pred_list = [t[2] for t in test_points]
    mae, rmse, mape = compute_metrics(y_true, y_pred_list)

    history = list(values)
    last_date = dates[-1]
    preds = []
    for _ in range(horizon_months):
        w = min(window, len(history))
        yhat = float(np.mean(history[-w:]))
        next_month = (pd.Timestamp(last_date) + pd.offsets.MonthBegin(1)).date()
        preds.append((next_month, yhat))
        history.append(yhat)
        last_date = next_month

    return {"mae": mae, "rmse": rmse, "mape": mape, "preds": preds, "test_points": test_points}


def exponential_smoothing_forecast(dates, values, horizon_months=6, alpha=0.3, test_size=6):
    n = len(values)
    values = list(map(float, values))
    test_size = min(test_size, max(1, n - 2))
    train_end = n - test_size

    # initialise on training window
    smoothed = values[0]
    for v in values[1:train_end]:
        smoothed = alpha * v + (1 - alpha) * smoothed

    test_points = []
    for i in range(test_size):
        idx = train_end + i
        y_pred = smoothed
        test_points.append((dates[idx], values[idx], y_pred))
        smoothed = alpha * values[idx] + (1 - alpha) * smoothed

    y_true = [t[1] for t in test_points]
    y_pred_list = [t[2] for t in test_points]
    mae, rmse, mape = compute_metrics(y_true, y_pred_list)

    # retrain on all data for final forecast
    smoothed = values[0]
    for v in values[1:]:
        smoothed = alpha * v + (1 - alpha) * smoothed

    last_date = dates[-1]
    preds = []
    for _ in range(horizon_months):
        next_month = (pd.Timestamp(last_date) + pd.offsets.MonthBegin(1)).date()
        preds.append((next_month, smoothed))
        last_date = next_month

    return {"mae": mae, "rmse": rmse, "mape": mape, "preds": preds, "test_points": test_points}

