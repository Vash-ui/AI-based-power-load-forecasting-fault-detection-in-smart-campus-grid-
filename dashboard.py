# src/visualization/dashboard.py
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import logging
from pathlib import Path

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    """Load processed data"""
    data_path = Path("data/processed/train.csv")
    try:
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        logger.info(f"Data loaded successfully from {data_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise


def create_app():
    """Create Dash application"""
    app = Dash(__name__)

    # Layout
    app.layout = html.Div([
        html.H1("Campus Power Load Dashboard"),
        dcc.Graph(id='load-forecast-plot'),
        dcc.Interval(
            id='interval-component',
            interval=60 * 1000,  # Update every minute
            n_intervals=0
        )
    ])

    # Callback for live updates
    @app.callback(
        Output('load-forecast-plot', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def update_graph(n):
        df = load_data()
        fig = px.line(
            df,
            x='timestamp',
            y='power_kw',
            color='building_id',
            title='Real-time Power Consumption'
        )
        return fig

    return app


if __name__ == '__main__':
    app = create_app()
    app.run_server(debug=True, host='0.0.0.0', port=8050)