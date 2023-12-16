import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

# Crear datos de ejemplo
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', end='2022-01-10', freq='D')
model_metrics = pd.DataFrame({
'Fecha': dates,
'Precision': np.random.uniform(0.8, 0.95, len(dates)),
'Recall': np.random.uniform(0.7, 0.9, len(dates)),
'Exactitud': np.random.uniform(0.85, 0.98, len(dates))})

model_metrics.to_csv('metricas.csv', index=False)


# Cargar datos
model_metrics = pd.read_csv('metricas.csv') # Cambiar la ruta
# Inicializar la aplicación Dash
app = dash.Dash(__name__)
# Diseño del layout
app.layout = html.Div([
html.H1("Dashboard de Monitoreo de Modelos de Datos"),
# Gráfico de Precisión
dcc.Graph(
id='precision-plot',
figure=px.line(model_metrics, x='Fecha', y='Precision', title='Precisión a lo largo del tiempo')),
# Gráfico de Recall
dcc.Graph(
id='recall-plot',
figure=px.line(model_metrics, x='Fecha', y='Recall', title='Recall a lo largo del tiempo')),
# Gráfico de Exactitud
dcc.Graph(
id='accuracy-plot',
figure=px.line(model_metrics, x='Fecha', y='Exactitud', title='Exactitud a lo largo del tiempo')),])
# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)