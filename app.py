"""
CRYPTOVIEW PRO - Sistema Avanzado de Pronóstico de Criptomonedas
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Imports de módulos
from config.settings import *
from data.collectors import CryptoDataCollector
from utils.indicators import TechnicalIndicators
from utils.alerts import alert_system

# Configuración de página
st.set_page_config(
    page_title="CryptoView Pro",
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

# ============ SIDEBAR ============
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/bitcoin.png", width=80)
    st.title("⚙️ Configuración")
    
    st.markdown("---")
    
    crypto_symbol = st.selectbox(
        "💰 Criptomoneda",
        AVAILABLE_CRYPTOS,
        index=AVAILABLE_CRYPTOS.index(DEFAULT_CRYPTO)
    )
    
    timeframe = st.selectbox(
        "⏱️ Timeframe",
        list(TIMEFRAMES.keys()),
        format_func=lambda x: f"{x} ({TIMEFRAMES[x]})",
        index=list(TIMEFRAMES.keys()).index(DEFAULT_TIMEFRAME)
    )
    
    forecast_hours = st.slider(
        "🔮 Horizonte de predicción (horas)",
        min_value=1,
        max_value=168,
        value=FORECAST_HOURS
    )
    
    st.markdown("---")
    st.markdown("### 📊 Vista de Datos")
    
    data_limit = st.number_input(
        "Cantidad de datos a cargar",
        min_value=100,
        max_value=2000,
        value=DATA_LIMIT,
        step=100
    )
    
    show_volume = st.checkbox("Mostrar volumen", value=True)
    
    st.markdown("---")
    st.caption("💡 CryptoView Pro v1.0")
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
    """Crea gráfico principal"""
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
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Precio',
                line=dict(color=COLORS['accent'], width=2)
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
                line=dict(color='orange', width=1, dash='dash')
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
                line=dict(color='purple', width=1, dash='dash')
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
        title=f"{crypto_symbol} - Análisis",
        xaxis_title="Fecha",
        yaxis_title="Precio (USDT)",
        hovermode='x unified',
        template='plotly_dark',
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False
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
            Sistema Avanzado de Pronóstico de Criptomonedas
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============ CARGAR DATOS ============
with st.spinner(f'📡 Cargando datos de {crypto_symbol}...'):
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
volatility = df['close'].pct_change().std() * 100

current_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else np.nan
signals = TechnicalIndicators.get_signals(df)
overall_signal = signals.get('overall', 'neutral')

price_color = COLORS['success'] if change_24h >= 0 else COLORS['danger']
signal_emoji = {'buy': '🟢', 'sell': '🔴', 'neutral': '🟡'}.get(overall_signal, '⚪')

# Display métricas
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 14px; color: #9CA3AF;'>💰 Precio Actual</div>
        <div class='big-metric'>${current_price:,.2f}</div>
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
        <div style='font-size: 14px; color: #9CA3AF;'>🎯 RSI (14)</div>
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
        <div style='font-size: 12px; color: #6B7280; text-transform: uppercase;'>{overall_signal}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============ TABS ============
tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "📊 Análisis Técnico", "🔔 Alertas"])

# TAB 1: DASHBOARD
with tab1:
    st.markdown("### 📈 Gráfico Principal")
    
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
    
    # Datos recientes
    st.markdown("### 📋 Datos Recientes")
    recent_data = df[['open', 'high', 'low', 'close', 'volume']].tail(10)
    st.dataframe(recent_data, use_container_width=True)

# TAB 2: ANÁLISIS TÉCNICO
with tab2:
    st.markdown("## 📊 Análisis Técnico Detallado")
    
    # RSI
    st.markdown("### 📉 RSI (Relative Strength Index)")
    
    if 'rsi' in df.columns:
        fig_rsi = go.Figure()
        
        fig_rsi.add_trace(go.Scatter(
            x=df.index,
            y=df['rsi'],
            mode='lines',
            name='RSI',
            line=dict(color=COLORS['accent'], width=2)
        ))
        
        fig_rsi.add_hline(y=70, line_dash="dash", line_color=COLORS['danger'], annotation_text="Sobrecompra")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color=COLORS['success'], annotation_text="Sobreventa")
        
        fig_rsi.update_layout(
            height=300,
            template='plotly_dark',
            showlegend=True,
            yaxis_title="RSI"
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    # MACD
    st.markdown("### 📈 MACD")
    
    if 'macd' in df.columns:
        fig_macd = go.Figure()
        
        fig_macd.add_trace(go.Scatter(
            x=df.index,
            y=df['macd'],
            mode='lines',
            name='MACD',
            line=dict(color='#4ECDC4', width=2)
        ))
        
        fig_macd.add_trace(go.Scatter(
            x=df.index,
            y=df['macd_signal'],
            mode='lines',
            name='Signal',
            line=dict(color='#FF6B6B', width=2)
        ))
        
        colors_hist = [COLORS['success'] if val >= 0 else COLORS['danger'] for val in df['macd_hist']]
        
        fig_macd.add_trace(go.Bar(
            x=df.index,
            y=df['macd_hist'],
            name='Histogram',
            marker_color=colors_hist,
            opacity=0.5
        ))
        
        fig_macd.update_layout(
            height=300,
            template='plotly_dark',
            showlegend=True
        )
        
        st.plotly_chart(fig_macd, use_container_width=True)
    
    # Tabla de indicadores
    st.markdown("### 📋 Indicadores Actuales")
    
    indicators_data = []
    
    if 'rsi' in df.columns:
        rsi_val = df['rsi'].iloc[-1]
        indicators_data.append({
            'Indicador': 'RSI (14)',
            'Valor': f'{rsi_val:.2f}',
            'Estado': '🔴 Sobrecompra' if rsi_val > 70 else '🟢 Sobreventa' if rsi_val < 30 else '🟡 Neutral'
        })
    
    if 'macd' in df.columns:
        macd_val = df['macd'].iloc[-1]
        macd_sig = df['macd_signal'].iloc[-1]
        indicators_data.append({
            'Indicador': 'MACD',
            'Valor': f'{macd_val:.4f}',
            'Estado': '🟢 Alcista' if macd_val > macd_sig else '🔴 Bajista'
        })
    
    df_indicators = pd.DataFrame(indicators_data)
    st.dataframe(df_indicators, use_container_width=True, hide_index=True)

# TAB 3: ALERTAS
with tab3:
    st.markdown("## 🔔 Sistema de Alertas")
    
    st.info("🚧 Sistema de alertas básico. Funcionalidad completa próximamente.")
    
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
        
        submitted = st.form_submit_button("✅ Crear Alerta", use_container_width=True)
        
        if submitted:
            new_alert = {
                'crypto': crypto_symbol,
                'type': alert_type,
                'condition': condition,
                'threshold': threshold,
                'created': datetime.now()
            }
            st.session_state.alerts.append(new_alert)
            st.success(f"✅ Alerta creada: {alert_type} {condition} {threshold}")
    
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
    st.caption("🚀 CryptoView Pro v1.0")
with col_f2:
    st.caption("💻 Desarrollado con ❤️ usando Streamlit")
with col_f3:
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
