# src/visualization/dashboard.py
import plotly.express as px
import dash

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='load-forecast'),
    dcc.Interval(id='refresh', interval=60*1000)
])

@app.callback(Output('load-forecast', 'figure'),
              Input('refresh', 'n_intervals'))
def update_graph(_):
    df = get_latest_data()
    return px.line(df, x='timestamp', y=['actual', 'predicted'])