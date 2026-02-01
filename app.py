"""
CRYPTOVIEW PRO - Sistema Avanzado de Pronóstico de Criptomonedas
Developed by Julian E. Coronado Gil - Data Scientist
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import warnings

warnings.filterwarnings('ignore')

# Imports de módulos
from config.settings import *
from data.collectors import CryptoDataCollector
from utils.indicators import TechnicalIndicators
from utils.alerts import alert_system
from utils.backtesting import Backtester

# Imports de modelos ML
ML_AVAILABLE = False
PROPHET_AVAILABLE = False
HYBRID_AVAILABLE = False

try:
    from models.xgboost_model import XGBoostCryptoPredictor, backtest_model, create_prediction_intervals
    ML_AVAILABLE = True
except ImportError:
    pass

try:
    from models.prophet_model import ProphetCryptoPredictor, backtest_prophet
    PROPHET_AVAILABLE = True
except ImportError:
    pass

try:
    from models.hybrid_model import HybridCryptoPredictor
    HYBRID_AVAILABLE = True
except ImportError:
    pass

# Configuración de página
st.set_page_config(
    page_title="CryptoView Pro - by Julian E. Coronado Gil",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Session State
if 'collector' not in st.session_state:
    st.session_state.collector = CryptoDataCollector('kraken')

if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()

if 'alerts' not in st.session_state:
    st.session_state.alerts = []

if 'predictor_xgb' not in st.session_state:
    if ML_AVAILABLE:
        st.session_state.predictor_xgb = XGBoostCryptoPredictor()
    else:
        st.session_state.predictor_xgb = None

if 'predictor_prophet' not in st.session_state:
    if PROPHET_AVAILABLE:
        st.session_state.predictor_prophet = ProphetCryptoPredictor()
    else:
        st.session_state.predictor_prophet = None

if 'predictor_hybrid' not in st.session_state:
    if HYBRID_AVAILABLE:
        st.session_state.predictor_hybrid = HybridCryptoPredictor()
    else:
        st.session_state.predictor_hybrid = None

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

if 'predictions' not in st.session_state:
    st.session_state.predictions = None

if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'hybrid'

# ============ SIDEBAR ============
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/bitcoin.png", width=80)
    st.title("⚙️ Configuración")
    
    st.markdown("---")
    
    # CRIPTOMONEDA
    st.markdown("### 💰 Criptomoneda")
    st.caption("📌 Selecciona qué activo digital quieres analizar")
    crypto_symbol = st.selectbox(
        "Elige una:",
        AVAILABLE_CRYPTOS,
        index=AVAILABLE_CRYPTOS.index(DEFAULT_CRYPTO),
        label_visibility="collapsed"
    )
    
    # TIMEFRAME
    st.markdown("### ⏱️ Marco Temporal")
    st.caption("📊 Intervalo de cada vela")
    timeframe = st.selectbox(
        "Intervalo:",
        list(TIMEFRAMES.keys()),
        format_func=lambda x: f"{x} ({TIMEFRAMES[x]})",
        index=list(TIMEFRAMES.keys()).index(DEFAULT_TIMEFRAME),
        label_visibility="collapsed"
    )
    
    # HORIZONTE DE PREDICCIÓN - MEJORADO
    st.markdown("### 🔮 Horizonte de Predicción")
    st.caption("🎯 Selecciona el período a predecir")
    
    forecast_preset = st.selectbox(
        "Período:",
        ["24 Horas (1 Día)", "72 Horas (3 Días)", "168 Horas (1 Semana)", 
         "336 Horas (2 Semanas)", "720 Horas (1 Mes)", "Personalizado"],
        index=2,  # Default: 1 semana
        label_visibility="collapsed"
    )
    
    # Mapeo de presets
    forecast_map = {
        "24 Horas (1 Día)": 24,
        "72 Horas (3 Días)": 72,
        "168 Horas (1 Semana)": 168,
        "336 Horas (2 Semanas)": 336,
        "720 Horas (1 Mes)": 720
    }
    
    if forecast_preset == "Personalizado":
        forecast_hours = st.slider(
            "Horas personalizadas:",
            min_value=6,
            max_value=720,
            value=168,
            step=6
        )
    else:
        forecast_hours = forecast_map[forecast_preset]
        st.info(f"📊 Predicción: **{forecast_hours} horas** hacia el futuro")
    
    # Recomendación de modelo según horizonte
    if forecast_hours <= 72:
        recommended_model = "🎯 **Recomendado:** XGBoost (corto plazo)"
    elif forecast_hours <= 168:
        recommended_model = "🔀 **Recomendado:** Híbrido (balance)"
    else:
        recommended_model = "📊 **Recomendado:** Prophet (largo plazo)"
    
    st.caption(recommended_model)
    
    st.markdown("---")
    st.markdown("### 🤖 Modelo de Predicción")
    
    model_options = []
    if ML_AVAILABLE:
        model_options.append("XGBoost (Corto Plazo)")
    if PROPHET_AVAILABLE:
        model_options.append("Prophet (Largo Plazo)")
    if HYBRID_AVAILABLE:
        model_options.append("Híbrido (Recomendado)")
    
    if model_options:
        selected_model_display = st.selectbox(
            "Selecciona modelo:",
            model_options,
            index=len(model_options)-1 if "Híbrido (Recomendado)" in model_options else 0,
            label_visibility="collapsed"
        )
        
        # Mapear selección a nombre interno
        if "XGBoost" in selected_model_display:
            st.session_state.selected_model = 'xgboost'
        elif "Prophet" in selected_model_display:
            st.session_state.selected_model = 'prophet'
        else:
            st.session_state.selected_model = 'hybrid'
    
    st.markdown("---")
    st.markdown("### 📊 Configuración de Datos")
    
    # CANTIDAD DE DATOS - AUMENTADO
    st.caption("💾 Datos históricos para entrenar el modelo")
    data_limit = st.number_input(
        "Puntos de datos:",
        min_value=500,
        max_value=2000,
        value=2000,  # ✅ Aumentado a 2000
        step=100,
        label_visibility="collapsed"
    )
    
    # Info dinámica
    timeframe_hours = {'1m': 1/60, '5m': 5/60, '15m': 15/60, '1h': 1, '4h': 4, '1d': 24}
    if timeframe in timeframe_hours:
        days_covered = int(data_limit * timeframe_hours[timeframe] / 24)
        if days_covered > 0:
            st.caption(f"📊 ~{days_covered} días de histórico (~{days_covered/30:.1f} meses)")
    
    # MOSTRAR VOLUMEN
    show_volume = st.checkbox("📈 Mostrar volumen", value=True)
    
    # BANDAS DE CONFIANZA
    show_confidence = st.checkbox("📊 Bandas de confianza", value=True)
    st.caption("🎯 Rango probable del precio")
    
    # COMPARAR MODELOS
    compare_models = st.checkbox("🔬 Comparar todos los modelos", value=False)
    st.caption("📊 Ver XGBoost, Prophet e Híbrido juntos")
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.caption("**Julian E. Coronado Gil**")
    st.caption("Data Scientist")
    st.markdown("---")
    st.caption("💡 CryptoView Pro v2.0")
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")

# ============ FUNCIONES ============

@st.cache_data(ttl=60, show_spinner=False)
def load_crypto_data(symbol, timeframe, limit):
    """Carga datos con caché"""
    collector = CryptoDataCollector('kraken')
    df = collector.fetch_ohlcv(symbol, timeframe, limit)
    
    if not df.empty:
        df = TechnicalIndicators.add_all_indicators(df)
    
    return df

def create_main_chart(df, show_volume=True):
    """Crea gráfico principal con zoom y controles"""
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=('Precio', 'Volumen de Transacciones')
        )
    else:
        fig = make_subplots(rows=1, cols=1)
    
    # Candlestick
    if 'open' in df.columns:
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Precio',
                increasing_line_color=COLORS['success'],
                decreasing_line_color=COLORS['danger']
            ),
            row=1, col=1
        )
    
    # EMAs
    if 'ema_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ema_50'],
                mode='lines',
                name='EMA 50',
                line=dict(color='orange', width=2, dash='dash')
            ),
            row=1, col=1
        )
    
    if 'ema_200' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ema_200'],
                mode='lines',
                name='EMA 200',
                line=dict(color='purple', width=2, dash='dash')
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_upper'],
                mode='lines',
                name='Banda Superior',
                line=dict(color='rgba(173, 204, 255, 0.5)', width=1),
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_lower'],
                mode='lines',
                name='Banda Inferior',
                line=dict(color='rgba(173, 204, 255, 0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(173, 204, 255, 0.1)',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Volumen
    if show_volume and 'volume' in df.columns:
        colors_vol = [COLORS['success'] if close >= open_price else COLORS['danger'] 
                     for close, open_price in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volumen',
                marker_color=colors_vol,
                opacity=0.5
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title=f"{crypto_symbol} - Análisis en Tiempo Real",
        xaxis_title="Fecha",
        yaxis_title="Precio (USDT)",
        hovermode='x unified',
        template='plotly_dark',
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.1,
                y=1.15,
                buttons=list([
                    dict(args=[{"xaxis.range": [df.index[-168], df.index[-1]]}],
                         label="7D", method="relayout"),
                    dict(args=[{"xaxis.range": [df.index[-48], df.index[-1]]}],
                         label="2D", method="relayout"),
                    dict(args=[{"xaxis.range": [df.index[-24], df.index[-1]]}],
                         label="24H", method="relayout"),
                    dict(args=[{"xaxis.range": [df.index[0], df.index[-1]]}],
                         label="Todo", method="relayout"),
                ])
            )
        ]
    )
    
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            bgcolor='rgba(150, 150, 150, 0.1)',
            font=dict(color='white')
        )
    )
    
    return fig

def create_prediction_chart(df, predictions_dict, show_confidence=True, compare_mode=False):
    """
    Crea gráfico de predicciones con múltiples modelos
    
    Args:
        df: DataFrame histórico
        predictions_dict: Dict con predicciones (puede tener 'xgboost', 'prophet', 'hybrid')
        show_confidence: Mostrar bandas
        compare_mode: Si True, muestra todos los modelos juntos
    """
    fig = go.Figure()
    
    # Contexto histórico
    context_points = min(336, len(df))
    historical_data = df.tail(context_points)
    
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data['close'],
            mode='lines',
            name='Precio Histórico',
            line=dict(color=COLORS['accent'], width=2),
            hovertemplate='<b>Histórico</b><br>Precio: $%{y:,.2f}<br>%{x}<extra></extra>'
        )
    )
    
    # Colores para cada modelo
    model_colors = {
        'xgboost': '#00ff88',  # Verde
        'prophet': '#ffd700',  # Dorado
        'hybrid': '#ff6b9d'    # Rosa
    }
    
    model_names = {
        'xgboost': 'XGBoost (Corto Plazo)',
        'prophet': 'Prophet (Largo Plazo)',
        'hybrid': 'Híbrido (Combinado)'
    }
    
    # Determinar qué mostrar
    if compare_mode and isinstance(predictions_dict, dict):
        # Mostrar todos los modelos disponibles
        models_to_show = [k for k in ['xgboost', 'prophet', 'hybrid'] if k in predictions_dict]
    else:
        # Mostrar solo el recomendado o principal
        if isinstance(predictions_dict, dict):
            if 'hybrid' in predictions_dict:
                models_to_show = ['hybrid']
            elif 'recommended' in predictions_dict:
                models_to_show = [predictions_dict['recommended']]
            else:
                models_to_show = list(predictions_dict.keys())[:1]
        else:
            models_to_show = ['main']
            predictions_dict = {'main': predictions_dict}
    
    # Dibujar predicciones
    for model_key in models_to_show:
        if model_key not in predictions_dict:
            continue
        
        pred_df = predictions_dict[model_key]
        color = model_colors.get(model_key, '#00d9ff')
        name = model_names.get(model_key, 'Predicción')
        
        # Línea principal
        fig.add_trace(
            go.Scatter(
                x=pred_df.index,
                y=pred_df['predicted_price'],
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=3, dash='dash' if compare_mode else 'solid'),
                marker=dict(size=4 if compare_mode else 6, symbol='diamond'),
                hovertemplate=f'<b>{name}</b><br>Precio: $%{{y:,.2f}}<br>%{{x}}<extra></extra>'
            )
        )
        
        # Bandas de confianza (solo si no es modo comparación o es el modelo principal)
        if show_confidence and 'lower_bound' in pred_df.columns and (not compare_mode or model_key == models_to_show[0]):
            fig.add_trace(
                go.Scatter(
                    x=pred_df.index,
                    y=pred_df['upper_bound'],
                    mode='lines',
                    name=f'{name} - Superior',
                    line=dict(color=color, width=0),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=pred_df.index,
                    y=pred_df['lower_bound'],
                    mode='lines',
                    name=f'{name} - Inferior',
                    line=dict(color=color, width=0),
                    fill='tonexty',
                    fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)',
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
    
    # Línea "Ahora"
    if len(df) > 0:
        last_timestamp = df.index[-1]
        last_price = df['close'].iloc[-1]
        
        fig.add_shape(
            type="line",
            x0=last_timestamp,
            x1=last_timestamp,
            y0=0,
            y1=1,
            yref='paper',
            line=dict(color="white", width=2, dash="dot"),
            opacity=0.7
        )
        
        fig.add_annotation(
            x=last_timestamp,
            y=1.02,
            yref='paper',
            text=f"⏰ Ahora: {last_timestamp.strftime('%Y-%m-%d %H:%M')}",
            showarrow=False,
            font=dict(color="white", size=11),
            bgcolor="rgba(0, 0, 0, 0.6)",
            bordercolor="white",
            borderwidth=1,
            borderpad=4
        )
        
        fig.add_annotation(
            x=last_timestamp,
            y=last_price,
            text=f"💰 ${last_price:,.2f}",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=0,
            font=dict(color="white", size=12),
            bgcolor="rgba(0, 217, 255, 0.8)",
            bordercolor="white",
            borderwidth=1,
            borderpad=4
        )
    
    # Calcular días
    days_shown = int(context_points / 24) if context_points >= 24 else 1
    
    # Obtener número de períodos de la primera predicción disponible
    first_pred = predictions_dict[models_to_show[0]]
    num_periods = len(first_pred)
    
    title_suffix = f" - {num_periods}h ({num_periods/24:.1f} días)"
    if compare_mode:
        title_suffix += " - Comparación de Modelos"
    
    fig.update_layout(
        title=f"🔮 Predicción de {crypto_symbol}{title_suffix}",
        xaxis_title="Fecha y Hora",
        yaxis_title="Precio (USDT)",
        hovermode='x unified',
        template='plotly_dark',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6H", step="hour", stepmode="backward"),
                dict(count=24, label="24H", step="hour", stepmode="backward"),
                dict(count=7, label="7D", step="day", stepmode="backward"),
                dict(count=14, label="14D", step="day", stepmode="backward"),
                dict(step="all", label="Todo")
            ]),
            bgcolor='rgba(150, 150, 150, 0.1)',
            font=dict(color='white')
        )
    )
    
    return fig

def create_backtest_chart(actual, predicted, title="Backtesting"):
    """Crea gráfico de comparación para backtesting"""
    fig = go.Figure()
    
    x_values = list(range(len(actual)))
    
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=actual,
            mode='lines',
            name='Precio Real',
            line=dict(color=COLORS['accent'], width=2),
            hovertemplate='<b>Real</b><br>$%{y:,.2f}<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=predicted,
            mode='lines',
            name='Predicción del Modelo',
            line=dict(color=COLORS['forecast'], width=2, dash='dash'),
            hovertemplate='<b>Predicción</b><br>$%{y:,.2f}<extra></extra>'
        )
    )
    
    error = np.abs(actual - predicted)
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=error,
            mode='lines',
            name='Error Absoluto',
            line=dict(color=COLORS['danger'], width=1),
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.1)',
            yaxis='y2',
            hovertemplate='<b>Error</b><br>$%{y:,.2f}<extra></extra>'
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Período",
        yaxis_title="Precio (USDT)",
        yaxis2=dict(
            title="Error ($)",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode='x unified',
        template='plotly_dark',
        height=500,
        showlegend=True
    )
    
    return fig

# ============ HEADER ============
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='background: linear-gradient(90deg, #00D9FF 0%, #FFD700 100%);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;
                   font-size: 48px;
                   font-weight: 900;'>
            🚀 CRYPTOVIEW PRO
        </h1>
        <p style='color: #9CA3AF; font-size: 16px;'>
            Sistema Avanzado de Pronóstico con Modelos Híbridos
        </p>
        <p style='color: #6B7280; font-size: 12px; margin-top: 8px;'>
            Developed by <strong style='color: #00D9FF;'>Julian E. Coronado Gil</strong> - Data Scientist
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============ CARGAR DATOS ============
with st.spinner(f'📡 Cargando {data_limit} puntos de {crypto_symbol}...'):
    df = load_crypto_data(crypto_symbol, timeframe, data_limit)

if df.empty:
    st.error("❌ No se pudieron cargar los datos. Verifica la conexión.")
    st.stop()

st.session_state.data = df

# ============ MÉTRICAS ============
current_price = df['close'].iloc[-1]
price_24h_ago = df['close'].iloc[-24] if len(df) >= 24 else df['close'].iloc[0]
change_24h = current_price - price_24h_ago
change_24h_pct = (change_24h / price_24h_ago) * 100

volume_24h = df['volume'].tail(24).sum() if 'volume' in df.columns else 0
current_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else np.nan
signals = TechnicalIndicators.get_signals(df)
overall_signal = signals.get('overall', 'neutral')

price_color = COLORS['success'] if change_24h >= 0 else COLORS['danger']
signal_emoji = {'buy': '🟢', 'sell': '🔴', 'neutral': '🟡'}.get(overall_signal, '⚪')
signal_text = {'buy': 'COMPRA', 'sell': 'VENTA', 'neutral': 'NEUTRAL'}.get(overall_signal, 'NEUTRAL')

# Display métricas
st.markdown("### 📊 Métricas Principales")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    last_update_time = df.index[-1]
    last_update_str = last_update_time.strftime('%Y-%m-%d')
    last_update_hour = last_update_time.strftime('%H:%M:%S')
    
    now = datetime.now(timezone.utc)
    if last_update_time.tzinfo is None:
        time_diff = now.replace(tzinfo=None) - last_update_time
    else:
        time_diff = now - last_update_time
    
    minutes_ago = int(time_diff.total_seconds() / 60)
    
    if minutes_ago < 1:
        time_ago_str = "Hace <1 min"
    elif minutes_ago < 60:
        time_ago_str = f"Hace {minutes_ago} min"
    else:
        hours_ago = int(minutes_ago / 60)
        time_ago_str = f"Hace {hours_ago}h"
    
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 14px; color: #9CA3AF;'>💰 Precio Actual</div>
        <div class='big-metric'>${current_price:,.2f}</div>
        <div style='font-size: 11px; color: #00D9FF; margin-top: 4px; font-weight: 600;'>
            📅 {last_update_str}
        </div>
        <div style='font-size: 11px; color: #9CA3AF; margin-top: 2px;'>
            🕐 {last_update_hour}
        </div>
        <div style='font-size: 10px; color: #6B7280; margin-top: 4px;'>
            ⏱️ {time_ago_str}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 14px; color: #9CA3AF;'>📈 Cambio 24h</div>
        <div style='font-size: 24px; font-weight: 700; color: {price_color};'>
            {change_24h_pct:+.2f}%
        </div>
        <div style='font-size: 12px; color: #6B7280;'>${change_24h:+,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 14px; color: #9CA3AF;'>📊 Volumen 24h</div>
        <div style='font-size: 20px; font-weight: 700; color: {COLORS['accent']};'>
            ${volume_24h/1e6:.1f}M
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    rsi_color = COLORS['danger'] if current_rsi > 70 else COLORS['success'] if current_rsi < 30 else COLORS['warning']
    rsi_status = "Sobrecompra" if current_rsi > 70 else "Sobreventa" if current_rsi < 30 else "Neutral"
    
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 14px; color: #9CA3AF;'>🎯 RSI</div>
        <div style='font-size: 24px; font-weight: 700; color: {rsi_color};'>
            {current_rsi:.1f}
        </div>
        <div style='font-size: 12px; color: #6B7280;'>{rsi_status}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 14px; color: #9CA3AF;'>🎲 Señal</div>
        <div style='font-size: 28px; font-weight: 700;'>{signal_emoji}</div>
        <div style='font-size: 14px; color: #6B7280; font-weight: 600;'>{signal_text}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============ TABS ============
if HYBRID_AVAILABLE or ML_AVAILABLE or PROPHET_AVAILABLE:
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Panel Principal", 
        "🔮 Predicciones ML", 
        "📊 Análisis Técnico", 
        "🔔 Alertas"
    ])
else:
    tab1, tab3, tab4 = st.tabs([
        "🏠 Panel Principal",
        "📊 Análisis Técnico",
        "🔔 Alertas"
    ])

# TAB 1: DASHBOARD
with tab1:
    st.markdown("### 📈 Gráfico de Velas con Indicadores")
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🔄 Actualizar Datos", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col_btn2:
        if st.button("📥 Descargar CSV", use_container_width=True):
            csv = df.to_csv()
            st.download_button(
                "⬇️ Descargar",
                csv,
                f"{crypto_symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    fig_main = create_main_chart(df, show_volume=show_volume)
    st.plotly_chart(fig_main, use_container_width=True)
    
    st.markdown("### 📋 Datos Recientes")
    recent_data = df[['open', 'high', 'low', 'close', 'volume']].tail(10)
    recent_data.columns = ['Apertura', 'Máximo', 'Mínimo', 'Cierre', 'Volumen']
    st.dataframe(recent_data, use_container_width=True)

# TAB 2: PREDICCIONES ML
if HYBRID_AVAILABLE or ML_AVAILABLE or PROPHET_AVAILABLE:
    with tab2:
        st.markdown("## 🔮 Predicciones con Machine Learning")
        
        # Mostrar qué modelos están disponibles
        available_models_list = []
        if ML_AVAILABLE:
            available_models_list.append("✅ XGBoost (Corto Plazo 1-3 días)")
        if PROPHET_AVAILABLE:
            available_models_list.append("✅ Prophet (Largo Plazo 1-4 semanas)")
        if HYBRID_AVAILABLE:
            available_models_list.append("✅ Híbrido (Combina ambos)")
        
        st.info("🤖 **Modelos disponibles:** " + " | ".join(available_models_list))
        
        if len(df) < 500:
            st.warning(f"⚠️ Se recomienda al menos 500 puntos. Tienes {len(df)}. Aumenta en la barra lateral.")
        
        col_ml1, col_ml2 = st.columns([2, 1])
        
        with col_ml1:
            with st.expander("ℹ️ ¿Cómo funcionan los modelos?"):
                st.markdown(f"""
                ### 🤖 Modelos de Predicción
                
                **XGBoost (Corto Plazo):**
                - Análisis técnico con 50+ features
                - Ideal para 1-3 días
                - Precisión: 65-75% dirección
                - MAPE típico: 3-7%
                
                **Prophet (Largo Plazo):**
                - Detecta tendencias y estacionalidad
                - Ideal para 1-4 semanas
                - Robusto ante volatilidad
                - Intervalos de confianza realistas
                
                **Híbrido (Recomendado):**
                - Combina XGBoost + Prophet
                - Peso dinámico según horizonte
                - Mejor precisión general
                - Adaptable a cualquier período
                
                **Datos actuales:**
                - Puntos históricos: {len(df)}
                - Período: ~{int(len(df) * timeframe_hours.get(timeframe, 1) / 24)} días
                - Horizonte: {forecast_hours} horas ({forecast_hours/24:.1f} días)
                """)
        
        with col_ml2:
            if st.button("🎯 Entrenar y Predecir", use_container_width=True, type="primary"):
                with st.spinner("🤖 Entrenando modelos..."):
                    try:
                        selected = st.session_state.selected_model
                        
                        if selected == 'hybrid' and HYBRID_AVAILABLE:
                            # Entrenar híbrido
                            training_info = st.session_state.predictor_hybrid.train(df)
                            st.session_state.model_trained = True
                            
                            # Predecir
                            predictions_dict = st.session_state.predictor_hybrid.predict_future(
                                df, 
                                periods=forecast_hours
                            )
                            
                            st.session_state.predictions = predictions_dict
                            st.session_state.backtest_results = {'model': 'hybrid', 'info': training_info}
                            
                        elif selected == 'xgboost' and ML_AVAILABLE:
                            # Solo XGBoost
                            backtest_results = backtest_model(df, st.session_state.predictor_xgb, train_size=0.8)
                            predictions = st.session_state.predictor_xgb.predict_future(df, periods=forecast_hours)
                            predictions = create_prediction_intervals(predictions)
                            
                            st.session_state.predictions = {'xgboost': predictions, 'recommended': 'xgboost'}
                            st.session_state.backtest_results = backtest_results
                            st.session_state.model_trained = True
                            
                        elif selected == 'prophet' and PROPHET_AVAILABLE:
                            # Solo Prophet
                            backtest_results = backtest_prophet(df, st.session_state.predictor_prophet, test_periods=min(168, forecast_hours))
                            predictions = st.session_state.predictor_prophet.predict_future(periods=forecast_hours, freq='H')
                            
                            st.session_state.predictions = {'prophet': predictions, 'recommended': 'prophet'}
                            st.session_state.backtest_results = backtest_results
                            st.session_state.model_trained = True
                        
                        st.success("✅ ¡Modelos entrenados y predicciones generadas!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("💡 Intenta aumentar los datos o cambiar el horizonte")
        
        # Mostrar predicciones
        if st.session_state.predictions is not None:
            st.markdown("---")
            st.markdown("### 📈 Predicciones Futuras")
            
            # Gráfico
            fig_pred = create_prediction_chart(
                df, 
                st.session_state.predictions,
                show_confidence=show_confidence,
                compare_mode=compare_models
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Resumen
            st.markdown("### 📊 Resumen de Predicciones")
            
            # Obtener la mejor predicción
            if isinstance(st.session_state.predictions, dict):
                if 'hybrid' in st.session_state.predictions:
                    best_pred = st.session_state.predictions['hybrid']
                    model_used = "Híbrido"
                elif 'recommended' in st.session_state.predictions:
                    rec = st.session_state.predictions['recommended']
                    best_pred = st.session_state.predictions[rec]
                    model_used = rec.upper()
                else:
                    best_pred = list(st.session_state.predictions.values())[0]
                    model_used = "ML"
            else:
                best_pred = st.session_state.predictions
                model_used = "ML"
            
            pred_current = current_price
            pred_final = best_pred['predicted_price'].iloc[-1]
            pred_change = pred_final - pred_current
            pred_change_pct = (pred_change / pred_current) * 100
            
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            
            with col_p1:
                st.metric("💰 Precio Actual", f"${pred_current:,.2f}")
            
            with col_p2:
                st.metric(
                    f"🔮 Predicción ({forecast_hours}h)",
                    f"${pred_final:,.2f}",
                    f"{pred_change_pct:+.2f}%"
                )
            
            with col_p3:
                pred_max = best_pred['predicted_price'].max()
                st.metric("📈 Máximo Previsto", f"${pred_max:,.2f}")
            
            with col_p4:
                pred_min = best_pred['predicted_price'].min()
                st.metric("📉 Mínimo Previsto", f"${pred_min:,.2f}")
            
            st.info(f"🤖 **Modelo usado:** {model_used}")
            
            # Tabla de predicciones
            with st.expander("📋 Ver tabla completa"):
                display_pred = best_pred.copy()
                display_pred['Fecha y Hora'] = display_pred.index.strftime('%Y-%m-%d %H:%M')
                display_pred['Precio'] = display_pred['predicted_price'].apply(lambda x: f"${x:,.2f}")
                
                if 'lower_bound' in display_pred.columns:
                    display_pred['Mínimo'] = display_pred['lower_bound'].apply(lambda x: f"${x:,.2f}")
                    display_pred['Máximo'] = display_pred['upper_bound'].apply(lambda x: f"${x:,.2f}")
                    cols = ['Fecha y Hora', 'Precio', 'Mínimo', 'Máximo']
                else:
                    cols = ['Fecha y Hora', 'Precio']
                
                st.dataframe(display_pred[cols].reset_index(drop=True), use_container_width=True, hide_index=True)
            
            # Backtesting
            if st.session_state.backtest_results:
                st.markdown("---")
                st.markdown("### 🧪 Evaluación del Modelo")
                
                results = st.session_state.backtest_results
                
                if 'model' in results and results['model'] == 'hybrid':
                    # Mostrar info de ambos modelos
                    st.markdown("#### Métricas de Entrenamiento")
                    
                    col_m1, col_m2 = st.columns(2)
                    
                    with col_m1:
                        st.markdown("**🎯 XGBoost:**")
                        xgb_info = results['info']['xgboost']
                        st.metric("MAPE", f"{xgb_info['test_mape']:.2f}%")
                        st.metric("Precisión Direccional", f"{xgb_info['test_direction_accuracy']:.2f}%")
                    
                    with col_m2:
                        st.markdown("**📊 Prophet:**")
                        prophet_info = results['info']['prophet']
                        st.metric("MAPE", f"{prophet_info['mape']:.2f}%")
                        st.metric("Precisión Direccional", f"{prophet_info['direction_accuracy']:.2f}%")
                
                else:
                    # Mostrar métricas normales
                    if 'metrics' in results:
                        metrics = results['metrics']
                        col_m1, col_m2 = st.columns(2)
                        
                        with col_m1:
                            st.markdown("#### Entrenamiento")
                            st.metric("MAE", f"${metrics['train_mae']:,.2f}")
                            st.metric("MAPE", f"{metrics['train_mape']:.2f}%")
                        
                        with col_m2:
                            st.markdown("#### Validación")
                            st.metric("MAE", f"${metrics['test_mae']:,.2f}")
                            st.metric("MAPE", f"{metrics['test_mape']:.2f}%")
                            st.metric("Precisión", f"{metrics['test_direction_accuracy']:.2f}%")
        
        else:
            st.info("👆 Click en 'Entrenar y Predecir' para generar predicciones")

# TAB 3: ANÁLISIS TÉCNICO
with tab3:
    st.markdown("## 📊 Análisis Técnico Detallado")
    
    # RSI
    st.markdown("### 📉 RSI")
    
    if 'rsi' in df.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['rsi'], mode='lines', name='RSI', line=dict(color=COLORS['accent'], width=2)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color=COLORS['danger'], annotation_text="Sobrecompra")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color=COLORS['success'], annotation_text="Sobreventa")
        fig_rsi.update_layout(height=300, template='plotly_dark', hovermode='x unified')
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    st.markdown("---")
    
    # MACD
    st.markdown("### 📈 MACD")
    
    if 'macd' in df.columns:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['macd'], mode='lines', name='MACD', line=dict(color='#4ECDC4', width=2)))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], mode='lines', name='Señal', line=dict(color='#FF6B6B', width=2)))
        
        colors_hist = [COLORS['success'] if val >= 0 else COLORS['danger'] for val in df['macd_hist']]
        fig_macd.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='Histograma', marker_color=colors_hist, opacity=0.5))
        fig_macd.update_layout(height=300, template='plotly_dark', hovermode='x unified')
        st.plotly_chart(fig_macd, use_container_width=True)

# TAB 4: ALERTAS
with tab4:
    st.markdown("## 🔔 Sistema de Alertas")
    st.info("🚧 Sistema de alertas en desarrollo")
    
    with st.form("alert_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            alert_type = st.selectbox("Tipo", ["Precio", "RSI", "Cambio %"])
            condition = st.selectbox("Condición", ["Mayor que", "Menor que"])
        
        with col2:
            if alert_type == "Precio":
                threshold = st.number_input("Valor", value=float(current_price), step=100.0)
            else:
                threshold = st.number_input("Valor", value=70.0, step=5.0)
        
        if st.form_submit_button("✅ Crear Alerta", use_container_width=True):
            st.session_state.alerts.append({
                'crypto': crypto_symbol,
                'type': alert_type,
                'condition': condition,
                'threshold': threshold,
                'created': datetime.now()
            })
            st.success(f"✅ Alerta creada")

# ============ FOOTER ============
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.caption("🚀 CryptoView Pro v2.0 Hybrid")
with col_f2:
    st.caption("👨‍💻 Developed by **Julian E. Coronado Gil** - Data Scientist")
with col_f3:
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 12px; padding: 20px;'>
    <p>⚠️ <strong>AVISO:</strong> Análisis técnico educativo. NO es asesoría financiera.</p>
</div>
""", unsafe_allow_html=True)
