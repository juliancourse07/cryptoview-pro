"""
Funciones de visualización reutilizables
"""
import plotly.graph_objects as go
import pandas as pd
from config.settings import COLORS

def create_candlestick_chart(df: pd.DataFrame, title: str = "Precio") -> go.Figure:
    """Crea gráfico de velas"""
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color=COLORS['success'],
        decreasing_line_color=COLORS['danger']
    )])
    
    fig.update_layout(
        title=title,
        template='plotly_dark',
        xaxis_rangeslider_visible=False
    )
    
    return fig
