# src/evaluation/evaluate.py
from sklearn.metrics import classification_report


def generate_reports(model, X_test, y_test):
    # Forecasting metrics
    forecast_metrics = {
        'MAE': mean_absolute_error(y_test, model.predict(X_test)),
        'RMSE': mean_squared_error(y_test, model.predict(X_test), squared=False)
    }

    # Anomaly detection metrics
    anomaly_report = classification_report(
        y_test_anomalies,
        model.predict_classes(X_test)
    )

    return forecast_metrics, anomaly_report