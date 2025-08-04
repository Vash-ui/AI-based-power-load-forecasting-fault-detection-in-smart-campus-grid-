# src/models/train_baseline.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import joblib
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(train_path: str, test_path: str):
    """Load training and test datasets"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    x_train = train.drop(columns=['target_column'])  # Replace with your actual target column
    y_train = train['target_column']
    x_test = test.drop(columns=['target_column'])
    y_test = test['target_column']

    return x_train, y_train, x_test, y_test


def train_baseline(x_train, y_train, **model_params):
    """Train baseline RandomForest model"""
    model = RandomForestRegressor(
        n_estimators=model_params.get('n_estimators', 100),
        random_state=42
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Evaluate model performance"""
    predictions = model.predict(x_test)

    metrics = {
        'MAE': mean_absolute_error(y_test, predictions),
        'RMSE': mean_squared_error(y_test, predictions, squared=False),
        'R2': r2_score(y_test, predictions)
    }

    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")

    return metrics


def save_model(model, output_dir="models"):
    """Save trained model to disk"""
    output_path = Path(output_dir) / "baseline_model.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(f"Model saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    try:
        # Load data - replace paths with your actual data
        x_train, y_train, x_test, y_test = load_data(
            train_path="data/processed/train.csv",
            test_path="data/processed/test.csv"
        )

        # Train model
        model = train_baseline(x_train, y_train, n_estimators=150)

        # Evaluate
        metrics = evaluate_model(model, x_test, y_test)

        # Save model
        save_model(model)

    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise