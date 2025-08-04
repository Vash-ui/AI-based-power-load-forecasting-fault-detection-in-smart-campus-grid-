# src/models/train_advanced.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import mlflow
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_preprocess_data(data_path: str):
    """Load and preprocess time-series data"""
    df = pd.read_csv(data_path, parse_dates=['timestamp'])

    # Feature engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Create sequences for LSTM
    def create_sequences(data, n_steps=24):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i + n_steps])
            y.append(data[i + n_steps])
        return np.array(X), np.array(y)

    X, y = create_sequences(df['power_kw'].values)  # Replace with your target column
    return X, y


def build_lstm_model(input_shape):
    """Build LSTM model architecture"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    return model


if __name__ == "__main__":
    try:
        # 1. Load data
        DATA_PATH = Path("/Users/vash/Desktop/AI-based Power Load Forecasting & Fault Detection in Smart Campus Grid/src/data_pipeline/create_train_csv.py")  # Update path
        X, y = load_and_preprocess_data(DATA_PATH)

        # 2. Split data
        x_train, x_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            shuffle=False  # Critical for time-series!
        )

        # 3. Train model
        with mlflow.start_run():
            model = build_lstm_model((x_train.shape[1], 1))

            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=50,
                batch_size=32
            )

            # Log metrics
            mlflow.log_metrics({
                "val_loss": history.history['val_loss'][-1],
                "train_loss": history.history['loss'][-1]
            })

            logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise