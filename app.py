"""
CRYPTOVIEW PRO - Sistema Avanzado de Pronóstico de Criptomonedas
Developed by Julian E. Coronado Gil - Data Scientist
Version 2.0 - Hybrid ML Models + Extended Predictions + Technical Analysis + Telegram Alerts
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

# Imports ML
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

# Imports utilidades predicciones
try:
    from utils.long_term_predictions import aggregate_predictions, create_calendar_view, get_milestone_predictions
    LONG_TERM_UTILS = True
except ImportError:
    LONG_TERM_UTILS = False

# Imports Telegram
try:
    from utils.telegram_notifier import TelegramNotifier
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

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

if 'telegram_notifier' not in st.session_state:
    if TELEGRAM_AVAILABLE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        st.session_state.telegram_notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    else:
        st.session_state.telegram_notifier = None

# ============ SIDEBAR ============
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/bitcoin.png", width=80)
    st.title("⚙️ Configuración")
    
    st.markdown("---")
    
    # CRIPTOMONEDA
    st.markdown("### 💰 Criptomoneda")
    st.caption("📌 Selecciona qué activo analizar")
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
    
    # HORIZONTE DE PREDICCIÓN - EXTENDIDO
    st.markdown("### 🔮 Horizonte de Predicción")
    st.caption("🎯 Selecciona hasta cuándo predecir")
    
    forecast_view = st.radio(
        "Vista:",
        ["📅 Corto Plazo", "📊 Mediano Plazo", "🎯 Largo Plazo", "🌟 Personalizado"],
        label_visibility="collapsed"
    )
    
    if forecast_view == "📅 Corto Plazo":
        st.caption("⚡ Trading intradía y swing")
        forecast_preset = st.selectbox(
            "Período:",
            ["6 Horas", "12 Horas", "24 Horas", "48 Horas", "72 Horas"],
            index=2
        )
        forecast_map = {
            "6 Horas": 6, "12 Horas": 12, "24 Horas": 24,
            "48 Horas": 48, "72 Horas": 72
        }
        forecast_hours = forecast_map[forecast_preset]
        
    elif forecast_view == "📊 Mediano Plazo":
        st.caption("📈 Posiciones de semanas")
        forecast_preset = st.selectbox(
            "Período:",
            ["1 Semana (168h)", "2 Semanas (336h)", "3 Semanas (504h)", "1 Mes (720h)"],
            index=0
        )
        forecast_map = {
            "1 Semana (168h)": 168, "2 Semanas (336h)": 336,
            "3 Semanas (504h)": 504, "1 Mes (720h)": 720
        }
        forecast_hours = forecast_map[forecast_preset]
        
    elif forecast_view == "🎯 Largo Plazo":
        st.caption("🎲 Largo plazo (mayor incertidumbre)")
        forecast_preset = st.selectbox(
            "Período:",
            ["2 Meses (1,440h)", "3 Meses (2,160h)", "6 Meses (4,320h)", "1 Año (8,760h)"],
            index=0
        )
        forecast_map = {
            "2 Meses (1,440h)": 1440, "3 Meses (2,160h)": 2160,
            "6 Meses (4,320h)": 4320, "1 Año (8,760h)": 8760
        }
        forecast_hours = forecast_map[forecast_preset]
        st.warning("⚠️ Predicciones >3 meses tienen mayor incertidumbre")
        
    else:  # Personalizado
        st.caption("⚙️ Define tu horizonte")
        col_custom1, col_custom2 = st.columns(2)
        with col_custom1:
            time_value = st.number_input("Cantidad:", min_value=1, max_value=365, value=7, step=1)
        with col_custom2:
            time_unit = st.selectbox("Unidad:", ["Horas", "Días", "Semanas", "Meses"])
        
        unit_map = {"Horas": 1, "Días": 24, "Semanas": 168, "Meses": 720}
        forecast_hours = time_value * unit_map[time_unit]
        st.info(f"📊 Total: **{forecast_hours}h** ({forecast_hours/24:.1f} días)")
    
    # Recomendación
    if forecast_hours <= 72:
        recommended_model = "🎯 Recomendado: XGBoost"
    elif forecast_hours <= 720:
        recommended_model = "🔀 Recomendado: Híbrido"
    else:
        recommended_model = "📊 Recomendado: Prophet"
    st.caption(recommended_model)
    
    st.markdown("---")
    st.markdown("### 🤖 Modelo")
    
    model_options = []
    if ML_AVAILABLE:
        model_options.append("XGBoost (Corto)")
    if PROPHET_AVAILABLE:
        model_options.append("Prophet (Largo)")
    if HYBRID_AVAILABLE:
        model_options.append("Híbrido ⭐")
    
    if model_options:
        selected_model_display = st.selectbox(
            "Modelo:",
            model_options,
            index=len(model_options)-1 if "Híbrido" in str(model_options) else 0,
            label_visibility="collapsed"
        )
        
        if "XGBoost" in selected_model_display:
            st.session_state.selected_model = 'xgboost'
        elif "Prophet" in selected_model_display:
            st.session_state.selected_model = 'prophet'
        else:
            st.session_state.selected_model = 'hybrid'
    
    st.markdown("---")
    st.markdown("### 📊 Datos")
    
    st.caption("💾 Datos históricos para entrenar")
    data_limit = st.number_input(
        "Puntos:",
        min_value=500,
        max_value=2000,
        value=2000,
        step=100,
        label_visibility="collapsed"
    )
    
    # Info dinámica
    timeframe_hours = {'1m': 1/60, '5m': 5/60, '15m': 15/60, '1h': 1, '4h': 4, '1d': 24}
    if timeframe in timeframe_hours:
        days_covered = int(data_limit * timeframe_hours[timeframe] / 24)
        if days_covered > 0:
            st.caption(f"📊 ~{days_covered} días (~{days_covered/30:.1f} meses)")
    
    show_volume = st.checkbox("📈 Volumen", value=True)
    show_confidence = st.checkbox("📊 Bandas confianza", value=True)
    compare_models = st.checkbox("🔬 Comparar modelos", value=False)
    
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
    """Gráfico principal"""
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=('Precio', 'Volumen')
        )
    else:
        fig = make_subplots(rows=1, cols=1)
    
    # Candlestick
    if 'open' in df.columns:
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df['open'], high=df['high'],
                low=df['low'], close=df['close'], name='Precio',
                increasing_line_color=COLORS['success'],
                decreasing_line_color=COLORS['danger']
            ), row=1, col=1
        )
    
    # EMAs
    if 'ema_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['ema_50'], mode='lines',
            name='EMA 50', line=dict(color='orange', width=2, dash='dash')
        ), row=1, col=1)
    
    if 'ema_200' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['ema_200'], mode='lines',
            name='EMA 200', line=dict(color='purple', width=2, dash='dash')
        ), row=1, col=1)
    
    # Bollinger
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['bb_upper'], mode='lines',
            name='BB Superior', line=dict(color='rgba(173,204,255,0.5)', width=1)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['bb_lower'], mode='lines',
            name='BB Inferior', line=dict(color='rgba(173,204,255,0.5)', width=1),
            fill='tonexty', fillcolor='rgba(173,204,255,0.1)'
        ), row=1, col=1)
    
    # Volumen
    if show_volume and 'volume' in df.columns:
        colors = [COLORS['success'] if c >= o else COLORS['danger'] 
                 for c, o in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(
            x=df.index, y=df['volume'], name='Volumen',
            marker_color=colors, opacity=0.5
        ), row=2, col=1)
    
    fig.update_layout(
        title=f"{crypto_symbol} - Análisis",
        xaxis_title="Fecha", yaxis_title="Precio (USDT)",
        hovermode='x unified', template='plotly_dark',
        height=700, showlegend=True, xaxis_rangeslider_visible=False,
        updatemenus=[dict(
            type="buttons", direction="left", x=0.1, y=1.15,
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
        )]
    )
    
    return fig

def create_prediction_chart(df, predictions_dict, show_confidence=True, compare_mode=False):
    """Gráfico de predicciones"""
    fig = go.Figure()
    
    context_points = min(336, len(df))
    historical_data = df.tail(context_points)
    
    fig.add_trace(go.Scatter(
        x=historical_data.index, y=historical_data['close'],
        mode='lines', name='Histórico',
        line=dict(color=COLORS['accent'], width=2),
        hovertemplate='<b>Histórico</b><br>$%{y:,.2f}<br>%{x}<extra></extra>'
    ))
    
    model_colors = {'xgboost': '#00ff88', 'prophet': '#ffd700', 'hybrid': '#ff6b9d'}
    model_names = {
        'xgboost': 'XGBoost', 'prophet': 'Prophet', 'hybrid': 'Híbrido'
    }
    
    if compare_mode and isinstance(predictions_dict, dict):
        models_to_show = [k for k in ['xgboost', 'prophet', 'hybrid'] if k in predictions_dict]
    else:
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
    
    for model_key in models_to_show:
        if model_key not in predictions_dict:
            continue
        
        pred_df = predictions_dict[model_key]
        color = model_colors.get(model_key, '#00d9ff')
        name = model_names.get(model_key, 'Predicción')
        
        fig.add_trace(go.Scatter(
            x=pred_df.index, y=pred_df['predicted_price'],
            mode='lines+markers', name=name,
            line=dict(color=color, width=3, dash='dash' if compare_mode else 'solid'),
            marker=dict(size=4 if compare_mode else 6, symbol='diamond'),
            hovertemplate=f'<b>{name}</b><br>$%{{y:,.2f}}<br>%{{x}}<extra></extra>'
        ))
        
        if show_confidence and 'lower_bound' in pred_df.columns and (not compare_mode or model_key == models_to_show[0]):
            fig.add_trace(go.Scatter(
                x=pred_df.index, y=pred_df['upper_bound'],
                mode='lines', name=f'{name} Superior',
                line=dict(color=color, width=0), showlegend=False, hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=pred_df.index, y=pred_df['lower_bound'],
                mode='lines', name=f'{name} Inferior',
                line=dict(color=color, width=0), fill='tonexty',
                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)',
                showlegend=False, hoverinfo='skip'
            ))
    
    # Línea "Ahora"
    if len(df) > 0:
        last_timestamp = df.index[-1]
        last_price = df['close'].iloc[-1]
        
        fig.add_shape(
            type="line", x0=last_timestamp, x1=last_timestamp,
            y0=0, y1=1, yref='paper',
            line=dict(color="white", width=2, dash="dot"), opacity=0.7
        )
        
        fig.add_annotation(
            x=last_timestamp, y=1.02, yref='paper',
            text=f"⏰ Ahora: {last_timestamp.strftime('%Y-%m-%d %H:%M')}",
            showarrow=False, font=dict(color="white", size=11),
            bgcolor="rgba(0,0,0,0.6)", bordercolor="white",
            borderwidth=1, borderpad=4
        )
        
        fig.add_annotation(
            x=last_timestamp, y=last_price, text=f"💰 ${last_price:,.2f}",
            showarrow=True, arrowhead=2, ax=40, ay=0,
            font=dict(color="white", size=12),
            bgcolor="rgba(0,217,255,0.8)", bordercolor="white",
            borderwidth=1, borderpad=4
        )
    
    days_shown = int(context_points / 24) if context_points >= 24 else 1
    first_pred = predictions_dict[models_to_show[0]]
    num_periods = len(first_pred)
    
    title_suffix = f" - {num_periods}h ({num_periods/24:.1f} días)"
    if compare_mode:
        title_suffix += " - Comparación"
    
    fig.update_layout(
        title=f"🔮 Predicción {crypto_symbol}{title_suffix}",
        xaxis_title="Fecha", yaxis_title="Precio (USDT)",
        hovermode='x unified', template='plotly_dark',
        height=600, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

def create_backtest_chart(actual, predicted, title="Backtesting"):
    """Gráfico de backtesting"""
    fig = go.Figure()
    
    x = list(range(len(actual)))
    
    fig.add_trace(go.Scatter(
        x=x, y=actual, mode='lines', name='Real',
        line=dict(color=COLORS['accent'], width=2),
        hovertemplate='<b>Real</b><br>$%{y:,.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=x, y=predicted, mode='lines', name='Predicción',
        line=dict(color=COLORS['forecast'], width=2, dash='dash'),
        hovertemplate='<b>Predicción</b><br>$%{y:,.2f}<extra></extra>'
    ))
    
    error = np.abs(actual - predicted)
    fig.add_trace(go.Scatter(
        x=x, y=error, mode='lines', name='Error',
        line=dict(color=COLORS['danger'], width=1),
        fill='tozeroy', fillcolor='rgba(255,68,68,0.1)',
        yaxis='y2', hovertemplate='<b>Error</b><br>$%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title, xaxis_title="Período", yaxis_title="Precio (USDT)",
        yaxis2=dict(title="Error ($)", overlaying='y', side='right', showgrid=False),
        hovermode='x unified', template='plotly_dark', height=500
    )
    
    return fig

# ============ HEADER ============
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='background: linear-gradient(90deg, #00D9FF 0%, #FFD700 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 48px; font-weight: 900;'>
            🚀 CRYPTOVIEW PRO
        </h1>
        <p style='color: #9CA3AF; font-size: 16px;'>
            Sistema Avanzado con Modelos Híbridos ML
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
    st.error("❌ No se pudieron cargar datos")
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
        time_ago_str = "<1 min"
    elif minutes_ago < 60:
        time_ago_str = f"{minutes_ago} min"
    else:
        time_ago_str = f"{int(minutes_ago/60)}h"
    
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 14px; color: #9CA3AF;'>💰 Precio Actual</div>
        <div class='big-metric'>${current_price:,.2f}</div>
        <div style='font-size: 11px; color: #00D9FF; margin-top: 4px;'>
            📅 {last_update_str}
        </div>
        <div style='font-size: 11px; color: #9CA3AF; margin-top: 2px;'>
            🕐 {last_update_hour}
        </div>
        <div style='font-size: 10px; color: #6B7280; margin-top: 4px;'>
            ⏱️ Hace {time_ago_str}
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
        "🏠 Panel", "🔮 Predicciones", "📊 Análisis Técnico", "🔔 Alertas"
    ])
else:
    tab1, tab3, tab4 = st.tabs(["🏠 Panel", "📊 Análisis Técnico", "🔔 Alertas"])

# TAB 1: PANEL
with tab1:
    st.markdown("### 📈 Gráfico Principal")
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🔄 Actualizar", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col_btn2:
        if st.button("📥 CSV", use_container_width=True):
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
    recent = df[['open', 'high', 'low', 'close', 'volume']].tail(10)
    recent.columns = ['Apertura', 'Máximo', 'Mínimo', 'Cierre', 'Volumen']
    st.dataframe(recent, use_container_width=True)

# TAB 2: PREDICCIONES
if HYBRID_AVAILABLE or ML_AVAILABLE or PROPHET_AVAILABLE:
    with tab2:
        st.markdown("## 🔮 Predicciones ML")
        
        available = []
        if ML_AVAILABLE:
            available.append("✅ XGBoost")
        if PROPHET_AVAILABLE:
            available.append("✅ Prophet")
        if HYBRID_AVAILABLE:
            available.append("✅ Híbrido")
        
        st.info("🤖 **Modelos:** " + " | ".join(available))
        
        if len(df) < 500:
            st.warning(f"⚠️ Tienes {len(df)} puntos. Recomendado: 500+")
        
        col_ml1, col_ml2 = st.columns([2, 1])
        
        with col_ml2:
            if st.button("🎯 Entrenar", use_container_width=True, type="primary"):
                with st.spinner("🤖 Entrenando..."):
                    try:
                        selected = st.session_state.selected_model
                        
                        if selected == 'hybrid' and HYBRID_AVAILABLE:
                            training_info = st.session_state.predictor_hybrid.train(df)
                            st.session_state.model_trained = True
                            predictions_dict = st.session_state.predictor_hybrid.predict_future(df, periods=forecast_hours)
                            st.session_state.predictions = predictions_dict
                            st.session_state.backtest_results = {'model': 'hybrid', 'info': training_info}
                            
                        elif selected == 'xgboost' and ML_AVAILABLE:
                            backtest_results = backtest_model(df, st.session_state.predictor_xgb, train_size=0.8)
                            predictions = st.session_state.predictor_xgb.predict_future(df, periods=forecast_hours)
                            predictions = create_prediction_intervals(predictions)
                            st.session_state.predictions = {'xgboost': predictions, 'recommended': 'xgboost'}
                            st.session_state.backtest_results = backtest_results
                            st.session_state.model_trained = True
                            
                        elif selected == 'prophet' and PROPHET_AVAILABLE:
                            backtest_results = backtest_prophet(df, st.session_state.predictor_prophet, test_periods=min(168, forecast_hours))
                            predictions = st.session_state.predictor_prophet.predict_future(periods=forecast_hours, freq='H')
                            st.session_state.predictions = {'prophet': predictions, 'recommended': 'prophet'}
                            st.session_state.backtest_results = backtest_results
                            st.session_state.model_trained = True
                        
                        st.success("✅ Listo!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Mostrar predicciones
        if st.session_state.predictions is not None:
            st.markdown("---")
            st.markdown("### 📈 Predicciones")
            
            fig_pred = create_prediction_chart(
                df, st.session_state.predictions,
                show_confidence=show_confidence,
                compare_mode=compare_models
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Obtener mejor predicción
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
            
            # Resumen
            st.markdown("### 📊 Resumen")
            
            pred_current = current_price
            pred_final = best_pred['predicted_price'].iloc[-1]
            pred_change = pred_final - pred_current
            pred_change_pct = (pred_change / pred_current) * 100
            
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            
            with col_p1:
                st.metric("💰 Actual", f"${pred_current:,.2f}")
            with col_p2:
                st.metric(f"🔮 Predicción ({forecast_hours}h)", f"${pred_final:,.2f}", f"{pred_change_pct:+.2f}%")
            with col_p3:
                st.metric("📈 Máximo", f"${best_pred['predicted_price'].max():,.2f}")
            with col_p4:
                st.metric("📉 Mínimo", f"${best_pred['predicted_price'].min():,.2f}")
            
            st.info(f"🤖 **Modelo:** {model_used}")
            
            # VISTAS AGREGADAS - LARGO PLAZO
            if forecast_hours > 168 and LONG_TERM_UTILS:
                st.markdown("---")
                st.markdown("### 📊 Vistas Agregadas (Largo Plazo)")
                
                tab_agg1, tab_agg2, tab_agg3 = st.tabs(["📅 Por Día", "📆 Por Semana", "🎯 Hitos"])
                
                with tab_agg1:
                    st.markdown("#### Predicción Diaria")
                    daily_agg = aggregate_predictions(best_pred, 'daily')
                    
                    fig_daily = go.Figure()
                    fig_daily.add_trace(go.Bar(
                        x=daily_agg['period'], y=daily_agg['avg_price'],
                        name='Precio Promedio', marker_color='lightblue'
                    ))
                    
                    fig_daily.update_layout(
                        title=f"Próximos {int(forecast_hours/24)} días",
                        xaxis_title="Fecha", yaxis_title="Precio",
                        template='plotly_dark', height=400
                    )
                    st.plotly_chart(fig_daily, use_container_width=True)
                
                with tab_agg2:
                    if forecast_hours >= 336:
                        st.markdown("#### Predicción Semanal")
                        weekly_agg = aggregate_predictions(best_pred, 'weekly')
                        
                        fig_weekly = go.Figure()
                        fig_weekly.add_trace(go.Scatter(
                            x=weekly_agg['period'], y=weekly_agg['avg_price'],
                            mode='lines+markers', name='Semanal',
                            line=dict(color='gold', width=3), marker=dict(size=10)
                        ))
                        
                        fig_weekly.update_layout(
                            title=f"Próximas {int(forecast_hours/168)} semanas",
                            template='plotly_dark', height=400
                        )
                        st.plotly_chart(fig_weekly, use_container_width=True)
                    else:
                        st.info("📊 Necesitas ≥2 semanas")
                
                with tab_agg3:
                    st.markdown("#### 🎯 Hitos Clave")
                    milestones = get_milestone_predictions(best_pred)
                    
                    if milestones:
                        milestone_keys = list(milestones.keys())
                        cols = st.columns(min(4, len(milestone_keys)))
                        
                        for i, (period, data) in enumerate(milestones.items()):
                            if i < len(cols):
                                with cols[i]:
                                    days = int(period.replace('d', ''))
                                    change = ((data['price'] - pred_current) / pred_current) * 100
                                    
                                    st.markdown(f"""
                                    <div class='metric-card'>
                                        <div style='font-size: 13px; color: #9CA3AF;'>📅 {days} días</div>
                                        <div style='font-size: 11px; color: #6B7280;'>{data['date'].strftime('%Y-%m-%d')}</div>
                                        <div style='font-size: 20px; font-weight: 700; color: {"#00ff88" if change > 0 else "#ff4444"}; margin-top: 8px;'>
                                            ${data['price']:,.2f}
                                        </div>
                                        <div style='font-size: 12px; color: #6B7280;'>{change:+.1f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
        
        else:
            st.info("👆 Click 'Entrenar' para predicciones")

# TAB 3: ANÁLISIS TÉCNICO (COMPLETO - CONTINUARÁ EN PRÓXIMA RESPUESTA SI ES NECESARIO)
# Por límite de caracteres, voy a simplificar esta sección

with tab3:
    st.markdown("## 📊 Análisis Técnico")
    
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
        
        colors = [COLORS['success'] if val >= 0 else COLORS['danger'] for val in df['macd_hist']]
        fig_macd.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='Histograma', marker_color=colors, opacity=0.5))
        fig_macd.update_layout(height=300, template='plotly_dark', hovermode='x unified')
        st.plotly_chart(fig_macd, use_container_width=True)

# TAB 4: ALERTAS + TELEGRAM
with tab4:
    st.markdown("## 🔔 Sistema de Alertas")
    
    if TELEGRAM_AVAILABLE and st.session_state.telegram_notifier:
        st.success("✅ Telegram conectado")
        
        col_test1, col_test2 = st.columns(2)
        with col_test1:
            if st.button("📱 Probar Conexión Telegram"):
                if st.session_state.telegram_notifier.test_connection():
                    st.success("✅ Mensaje de prueba enviado")
                else:
                    st.error("❌ Error al enviar")
    else:
        st.warning("⚠️ Telegram no configurado. Agrega TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID al .env")
    
    st.markdown("---")
    
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
            new_alert = {
                'crypto': crypto_symbol,
                'type': alert_type,
                'condition': condition,
                'threshold': threshold,
                'created': datetime.now()
            }
            st.session_state.alerts.append(new_alert)
            st.success(f"✅ Alerta creada")
            
            # Enviar a Telegram si está disponible
            if st.session_state.telegram_notifier:
                st.session_state.telegram_notifier.send_alert(
                    crypto_symbol, alert_type,
                    current_price if alert_type == "Precio" else current_rsi,
                    threshold, condition
                )
    
    if st.session_state.alerts:
        st.markdown("### 📋 Alertas Activas")
        for i, alert in enumerate(st.session_state.alerts):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{alert['crypto']}** - {alert['type']} {alert['condition']} {alert['threshold']}")
            with col2:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.alerts.pop(i)
                    st.rerun()

# ============ FOOTER ============
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.caption("🚀 CryptoView Pro v2.0")
with col_f2:
    st.caption("👨‍💻 Developed by **Julian E. Coronado Gil** - Data Scientist")
with col_f3:
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 12px; padding: 20px;'>
    <p>⚠️ Análisis técnico educativo. NO es asesoría financiera.</p>
</div>
""", unsafe_allow_html=True)
