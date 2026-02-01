"""
CRYPTOVIEW PRO - Sistema Avanzado de Pronóstico de Criptomonedas
Developed by Julian E. Coronado Gil - Data Scientist
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
from utils.backtesting import Backtester

# Imports de modelos ML
try:
    from models.xgboost_model import XGBoostCryptoPredictor, backtest_model
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

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

if 'predictor' not in st.session_state:
    if ML_AVAILABLE:
        st.session_state.predictor = XGBoostCryptoPredictor()
    else:
        st.session_state.predictor = None

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

if 'predictions' not in st.session_state:
    st.session_state.predictions = None

if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

# ============ SIDEBAR ============
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/bitcoin.png", width=80)
    st.title("⚙️ Configuración")
    
    st.markdown("---")
    
    # CRIPTOMONEDA con explicación
    st.markdown("### 💰 Criptomoneda")
    st.caption("📌 Selecciona qué activo digital quieres analizar")
    crypto_symbol = st.selectbox(
        "Elige una:",
        AVAILABLE_CRYPTOS,
        index=AVAILABLE_CRYPTOS.index(DEFAULT_CRYPTO),
        label_visibility="collapsed"
    )
    
    # TIMEFRAME con explicación
    st.markdown("### ⏱️ Marco Temporal")
    st.caption("📊 Intervalo de cada vela. Ejemplo: 1h = cada punto representa 1 hora de trading")
    timeframe = st.selectbox(
        "Intervalo:",
        list(TIMEFRAMES.keys()),
        format_func=lambda x: f"{x} ({TIMEFRAMES[x]})",
        index=list(TIMEFRAMES.keys()).index(DEFAULT_TIMEFRAME),
        label_visibility="collapsed"
    )
    
    # HORIZONTE DE PREDICCIÓN
    st.markdown("### 🔮 Horizonte de Predicción")
    st.caption("🎯 Horas hacia el futuro para predecir")
    forecast_hours = st.slider(
        "Horas:",
        min_value=6,
        max_value=168,
        value=24,
        step=6,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Configuración de Datos")
    
    # CANTIDAD DE DATOS
    st.caption("💾 Cantidad de puntos históricos. Más datos = análisis más completo")
    data_limit = st.number_input(
        "Puntos de datos:",
        min_value=200,
        max_value=2000,
        value=1000,
        step=100,
        label_visibility="collapsed"
    )
    
    # MOSTRAR VOLUMEN
    show_volume = st.checkbox("📈 Mostrar volumen de transacciones", value=True)
    st.caption("💡 El volumen indica cuánto dinero se mueve")
    
    # BANDAS DE CONFIANZA
    show_confidence = st.checkbox("📊 Mostrar bandas de confianza en predicciones", value=True)
    st.caption("🎯 Rango donde probablemente estará el precio")
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.caption("**Julian E. Coronado Gil**")
    st.caption("Data Scientist")
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
                name='EMA 50 (Media Móvil Rápida)',
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
                name='EMA 200 (Tendencia General)',
                line=dict(color='purple', width=2, dash='dash')
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        # Banda superior
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
        
        # Banda inferior
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
    
    # Layout con controles interactivos
    fig.update_layout(
        title=f"{crypto_symbol} - Análisis en Tiempo Real",
        xaxis_title="Fecha",
        yaxis_title="Precio (USDT)",
        hovermode='x unified',
        template='plotly_dark',
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        # Botones de zoom
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
    
    # Configurar zoom y pan
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            bgcolor='rgba(150, 150, 150, 0.1)',
            font=dict(color='white')
        )
    )
    
    return fig

def create_prediction_chart(df, predictions_df, show_confidence=True):
    """
    Crea gráfico de predicciones con bandas de confianza
    
    Args:
        df: DataFrame histórico
        predictions_df: DataFrame con predicciones
        show_confidence: Mostrar bandas de confianza
    """
    fig = go.Figure()
    
    # Datos históricos (últimos 168 puntos para contexto)
    historical_data = df.tail(168)
    
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
    
    # Predicciones
    fig.add_trace(
        go.Scatter(
            x=predictions_df.index,
            y=predictions_df['predicted_price'],
            mode='lines+markers',
            name='Predicción ML',
            line=dict(color=COLORS['forecast'], width=3, dash='dash'),
            marker=dict(size=6, symbol='diamond'),
            hovertemplate='<b>Predicción</b><br>Precio: $%{y:,.2f}<br>%{x}<extra></extra>'
        )
    )
    
    # Bandas de confianza
    if show_confidence and 'lower_bound' in predictions_df.columns:
        # Banda superior
        fig.add_trace(
            go.Scatter(
                x=predictions_df.index,
                y=predictions_df['upper_bound'],
                mode='lines',
                name='Límite Superior (95%)',
                line=dict(color='rgba(255, 215, 0, 0.3)', width=1),
                showlegend=True,
                hovertemplate='<b>Límite Superior</b><br>$%{y:,.2f}<extra></extra>'
            )
        )
        
        # Banda inferior con relleno
        fig.add_trace(
            go.Scatter(
                x=predictions_df.index,
                y=predictions_df['lower_bound'],
                mode='lines',
                name='Límite Inferior (95%)',
                line=dict(color='rgba(255, 215, 0, 0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(255, 215, 0, 0.15)',
                showlegend=True,
                hovertemplate='<b>Límite Inferior</b><br>$%{y:,.2f}<extra></extra>'
            )
        )
    
    # Línea vertical separando histórico de predicción
    if len(df) > 0:
        fig.add_vline(
            x=df.index[-1],
            line_dash="dot",
            line_color="white",
            opacity=0.5,
            annotation_text="Ahora",
            annotation_position="top"
        )
    
    fig.update_layout(
        title=f"🔮 Predicción de {crypto_symbol} - Próximas {len(predictions_df)} horas",
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
    
    # Agregar zoom y controles
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=24, label="24H", step="hour", stepmode="backward"),
                dict(count=7, label="7D", step="day", stepmode="backward"),
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
    
    # Valores reales
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
    
    # Predicciones
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
    
    # Área de error
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
            Sistema Avanzado de Pronóstico de Criptomonedas
        </p>
        <p style='color: #6B7280; font-size: 12px; margin-top: 8px;'>
            Developed by <strong style='color: #00D9FF;'>Julian E. Coronado Gil</strong> - Data Scientist
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============ CARGAR DATOS ============
with st.spinner(f'📡 Cargando datos de {crypto_symbol}...'):
    df = load_crypto_data(crypto_symbol, timeframe, data_limit)

if df.empty:
    st.error("❌ No se pudieron cargar los datos. Verifica la conexión con el exchange.")
    st.info("💡 **Tip:** Intenta cambiar de criptomoneda o refrescar la página.")
    st.stop()

st.session_state.data = df

# ============ MÉTRICAS CON EXPLICACIONES ============
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
signal_text = {'buy': 'COMPRA', 'sell': 'VENTA', 'neutral': 'NEUTRAL'}.get(overall_signal, 'NEUTRAL')

# Display métricas
st.markdown("### 📊 Métricas Principales")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 14px; color: #9CA3AF;'>💰 Precio Actual</div>
        <div class='big-metric'>${current_price:,.2f}</div>
        <div style='font-size: 11px; color: #6B7280; margin-top: 4px;'>
            Último precio registrado
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
        <div style='font-size: 11px; color: #6B7280; margin-top: 4px;'>
            Variación últimas 24h
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 14px; color: #9CA3AF;'>📊 Volumen 24h</div>
        <div style='font-size: 20px; font-weight: 700; color: {COLORS['accent']};'>
            ${volume_24h/1e6:.1f}M
        </div>
        <div style='font-size: 11px; color: #6B7280; margin-top: 4px;'>
            Total transaccionado
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    rsi_color = COLORS['danger'] if current_rsi > 70 else COLORS['success'] if current_rsi < 30 else COLORS['warning']
    rsi_status = "Sobrecompra" if current_rsi > 70 else "Sobreventa" if current_rsi < 30 else "Neutral"
    rsi_explanation = "Posible caída" if current_rsi > 70 else "Posible subida" if current_rsi < 30 else "Sin extremos"
    
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 14px; color: #9CA3AF;'>🎯 RSI</div>
        <div style='font-size: 24px; font-weight: 700; color: {rsi_color};'>
            {current_rsi:.1f}
        </div>
        <div style='font-size: 12px; color: #6B7280;'>{rsi_status}</div>
        <div style='font-size: 11px; color: #6B7280; margin-top: 4px;'>
            {rsi_explanation}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    signal_description = {
        'buy': 'Oportunidad de compra',
        'sell': 'Considerar venta',
        'neutral': 'Esperar'
    }.get(overall_signal, 'Analizando...')
    
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 14px; color: #9CA3AF;'>🎲 Señal</div>
        <div style='font-size: 28px; font-weight: 700;'>{signal_emoji}</div>
        <div style='font-size: 14px; color: #6B7280; font-weight: 600;'>{signal_text}</div>
        <div style='font-size: 11px; color: #6B7280; margin-top: 4px;'>
            {signal_description}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============ TABS ============
if ML_AVAILABLE:
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
    st.markdown("### 📈 Gráfico de Velas con Bandas de Bollinger")
    
    with st.expander("📖 ¿Cómo usar los controles del gráfico?"):
        st.markdown("""
        ### 🎮 Controles Interactivos
        
        **Botones de período:**
        - **7D, 2D, 24H, Todo:** Cambian el rango de tiempo visible
        
        **Controles del mouse:**
        - **Zoom:** Rueda del mouse o pinch en móvil
        - **Pan:** Click y arrastra para moverte
        - **Doble click:** Reset del zoom
        
        **Hover (pasar el mouse):**
        - Muestra todos los valores en ese punto
        - Fecha, precio, volumen, indicadores
        
        **Leyenda:**
        - Click en un elemento para ocultarlo/mostrarlo
        - Útil para ver solo lo que te interesa
        
        **📊 Bandas de Bollinger:**
        - Área azul sombreada alrededor del precio
        - Precio cerca de banda superior = sobrecompra
        - Precio cerca de banda inferior = sobreventa
        - Precio fuera de bandas = movimiento extremo
        """)
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🔄 Actualizar Datos", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col_btn2:
        if st.button("📥 Descargar Datos CSV", use_container_width=True):
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
    st.markdown("### 📋 Datos Recientes (Últimas 10 Velas)")
    recent_data = df[['open', 'high', 'low', 'close', 'volume']].tail(10)
    recent_data.columns = ['Apertura', 'Máximo', 'Mínimo', 'Cierre', 'Volumen']
    st.dataframe(recent_data, use_container_width=True)

# TAB 2: PREDICCIONES ML
if ML_AVAILABLE:
    with tab2:
        st.markdown("## 🔮 Predicciones con Machine Learning")
        st.caption("Modelo XGBoost entrenado con datos históricos y más de 50 features")
        
        if len(df) < 200:
            st.warning("⚠️ Se necesitan al menos 200 puntos de datos para entrenar el modelo. Aumenta el límite de datos en la barra lateral.")
        else:
            col_ml1, col_ml2 = st.columns([2, 1])
            
            with col_ml1:
                with st.expander("ℹ️ ¿Cómo funciona el modelo de predicción?"):
                    st.markdown("""
                    ### 🤖 Modelo XGBoost
                    
                    **¿Qué es?**
                    - Algoritmo de Machine Learning de última generación
                    - Usado por equipos ganadores de Kaggle
                    - Excelente para series temporales financieras
                    
                    **¿Qué analiza?**
                    - Más de 50 features técnicos
                    - Retornos en múltiples ventanas (1h, 4h, 24h, 7d)
                    - Medias móviles (7, 14, 30, 50)
                    - Volatilidad histórica
                    - RSI, MACD, Bollinger Bands
                    - Ratios de precio OHLC
                    - Patrones temporales (hora, día)
                    
                    **¿Qué tan preciso es?**
                    - MAPE típico: 2-8% (error porcentual)
                    - Precisión direccional: 65-75%
                    - Verifica las métricas de backtesting abajo
                    
                    **⚠️ Importante:**
                    - NO es 100% preciso (nadie lo es)
                    - Usa como herramienta, no como garantía
                    - Combina con análisis técnico
                    - Siempre gestiona tu riesgo
                    """)
            
            with col_ml2:
                if st.button("🎯 Entrenar Modelo y Predecir", use_container_width=True, type="primary"):
                    with st.spinner("🤖 Entrenando modelo con datos históricos..."):
                        try:
                            # Entrenar modelo
                            backtest_results = backtest_model(
                                df, 
                                st.session_state.predictor,
                                train_size=0.8
                            )
                            
                            st.session_state.backtest_results = backtest_results
                            st.session_state.model_trained = True
                            
                            # Generar predicciones
                            with st.spinner("🔮 Generando predicciones futuras..."):
                                predictions = st.session_state.predictor.predict_future(
                                    df, 
                                    periods=forecast_hours
                                )
                                
                                # Agregar intervalos de confianza
                                from models.xgboost_model import create_prediction_intervals
                                predictions = create_prediction_intervals(predictions)
                                
                                st.session_state.predictions = predictions
                            
                            st.success("✅ ¡Modelo entrenado y predicciones generadas exitosamente!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            st.info("💡 Intenta aumentar la cantidad de datos en la configuración")
            
            # Mostrar predicciones si existen
            if st.session_state.predictions is not None and not st.session_state.predictions.empty:
                st.markdown("---")
                st.markdown("### 📈 Predicciones Futuras")
                
                # Gráfico de predicciones
                fig_pred = create_prediction_chart(
                    df, 
                    st.session_state.predictions,
                    show_confidence=show_confidence
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Resumen de predicciones
                st.markdown("### 📊 Resumen de Predicciones")
                
                pred_df = st.session_state.predictions.copy()
                pred_current = current_price
                pred_final = pred_df['predicted_price'].iloc[-1]
                pred_change = pred_final - pred_current
                pred_change_pct = (pred_change / pred_current) * 100
                
                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                
                with col_p1:
                    st.metric(
                        "💰 Precio Actual",
                        f"${pred_current:,.2f}"
                    )
                
                with col_p2:
                    st.metric(
                        f"🔮 Predicción ({forecast_hours}h)",
                        f"${pred_final:,.2f}",
                        f"{pred_change_pct:+.2f}%"
                    )
                
                with col_p3:
                    pred_max = pred_df['predicted_price'].max()
                    st.metric(
                        "📈 Máximo Previsto",
                        f"${pred_max:,.2f}"
                    )
                
                with col_p4:
                    pred_min = pred_df['predicted_price'].min()
                    st.metric(
                        "📉 Mínimo Previsto",
                        f"${pred_min:,.2f}"
                    )
                
                # Tabla de predicciones
                with st.expander("📋 Ver tabla completa de predicciones"):
                    display_pred = pred_df.copy()
                    display_pred['Fecha y Hora'] = display_pred.index
                    display_pred['Precio Predicho'] = display_pred['predicted_price'].apply(lambda x: f"${x:,.2f}")
                    
                    if 'lower_bound' in display_pred.columns:
                        display_pred['Límite Inferior'] = display_pred['lower_bound'].apply(lambda x: f"${x:,.2f}")
                        display_pred['Límite Superior'] = display_pred['upper_bound'].apply(lambda x: f"${x:,.2f}")
                        display_cols = ['Fecha y Hora', 'Precio Predicho', 'Límite Inferior', 'Límite Superior']
                    else:
                        display_cols = ['Fecha y Hora', 'Precio Predicho']
                    
                    st.dataframe(
                        display_pred[display_cols].reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True
                    )
            
            # Mostrar backtesting si existe
            if st.session_state.backtest_results is not None:
                st.markdown("---")
                st.markdown("### 🧪 Backtesting - Evaluación del Modelo")
                st.caption("Comparación de predicciones del modelo vs precios reales en datos históricos")
                
                with st.expander("ℹ️ ¿Qué es el backtesting y cómo interpretarlo?"):
                    st.markdown("""
                    ### 🧪 Backtesting
                    
                    **¿Qué es?**
                    - Prueba del modelo con datos históricos reales
                    - Muestra cómo habría predicho en el pasado
                    - Indica qué tan preciso es el modelo
                    
                    **Métricas clave:**
                    
                    **MAE (Error Absoluto Medio):**
                    - Diferencia promedio entre predicción y realidad
                    - Más bajo = mejor
                    - Ejemplo: MAE de $500 = se equivoca en promedio $500
                    
                    **MAPE (Error Porcentual):**
                    - MAE expresado como porcentaje
                    - 2-5% = Excelente
                    - 5-10% = Bueno
                    - >10% = Mejorable
                    
                    **Precisión de Dirección:**
                    - % de veces que acertó si sube o baja
                    - >70% = Excelente
                    - 60-70% = Bueno
                    - <60% = Similar a azar (50%)
                    
                    **R² Score:**
                    - Qué tan bien explica la variación del precio
                    - 1.0 = Perfecto
                    - 0.8-1.0 = Excelente
                    - 0.5-0.8 = Bueno
                    - <0.5 = Limitado
                    """)
                
                results = st.session_state.backtest_results
                
                # Métricas de evaluación
                col_m1, col_m2 = st.columns(2)
                
                with col_m1:
                    st.markdown("#### 📊 Métricas de Entrenamiento")
                    metrics_train = results['metrics']
                    
                    st.metric("MAE", f"${metrics_train['train_mae']:,.2f}")
                    st.metric("MAPE", f"{metrics_train['train_mape']:.2f}%")
                    st.metric("Precisión Direccional", f"{metrics_train['train_direction_accuracy']:.2f}%")
                
                with col_m2:
                    st.markdown("#### 🎯 Métricas de Validación")
                    st.metric("MAE", f"${metrics_train['test_mae']:,.2f}")
                    st.metric("MAPE", f"{metrics_train['test_mape']:.2f}%")
                    st.metric("Precisión Direccional", f"{metrics_train['test_direction_accuracy']:.2f}%")
                
                # Gráficos de comparación
                st.markdown("#### 📈 Comparación: Predicción vs Realidad")
                
                tab_bt1, tab_bt2 = st.tabs(["Datos de Validación", "Datos de Entrenamiento"])
                
                with tab_bt1:
                    fig_test = create_backtest_chart(
                        results['test_actual'],
                        results['test_predicted'],
                        "Backtesting en Datos de Validación"
                    )
                    st.plotly_chart(fig_test, use_container_width=True)
                
                with tab_bt2:
                    fig_train = create_backtest_chart(
                        results['train_actual'],
                        results['train_predicted'],
                        "Backtesting en Datos de Entrenamiento"
                    )
                    st.plotly_chart(fig_train, use_container_width=True)
                
                # Feature importance
                with st.expander("🔍 Ver importancia de features"):
                    st.markdown("#### Top 20 Features más importantes")
                    st.caption("Estos son los factores que más influyen en las predicciones del modelo")
                    
                    importance_df = results['feature_importance'].head(20)
                    
                    fig_importance = go.Figure(go.Bar(
                        x=importance_df['importance'],
                        y=importance_df['feature'],
                        orientation='h',
                        marker_color=COLORS['accent']
                    ))
                    
                    fig_importance.update_layout(
                        title="Importancia de Features",
                        xaxis_title="Importancia",
                        yaxis_title="Feature",
                        template='plotly_dark',
                        height=600,
                        yaxis={'categoryorder':'total ascending'}
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
            
            else:
                st.info("👆 Haz click en 'Entrenar Modelo y Predecir' para generar predicciones")

# TAB 3: ANÁLISIS TÉCNICO
with tab3:
    st.markdown("## 📊 Análisis Técnico Detallado")
    st.caption("Indicadores avanzados para decisiones informadas")
    
    # RSI
    st.markdown("### 📉 RSI - Índice de Fuerza Relativa")
    
    with st.expander("ℹ️ ¿Qué es el RSI?"):
        st.markdown("""
        ### 🎯 RSI (Relative Strength Index)
        
        **¿Qué mide?**
        - Velocidad y magnitud de cambios de precio
        - Identifica sobrecompra/sobreventa
        
        **Interpretación:**
        - **RSI > 70:** Sobrecompra - Posible corrección bajista
        - **RSI < 30:** Sobreventa - Posible rebote alcista
        - **RSI 40-60:** Neutral
        """)
    
    if 'rsi' in df.columns:
        fig_rsi = go.Figure()
        
        fig_rsi.add_trace(go.Scatter(
            x=df.index,
            y=df['rsi'],
            mode='lines',
            name='RSI',
            line=dict(color=COLORS['accent'], width=2)
        ))
        
        fig_rsi.add_hline(y=70, line_dash="dash", line_color=COLORS['danger'], annotation_text="Sobrecompra (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color=COLORS['success'], annotation_text="Sobreventa (30)")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="#666", annotation_text="Línea Media (50)")
        
        fig_rsi.update_layout(
            height=300,
            template='plotly_dark',
            showlegend=True,
            yaxis_title="RSI",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    st.markdown("---")
    
    # MACD
    st.markdown("### 📈 MACD - Convergencia/Divergencia")
    
    with st.expander("ℹ️ ¿Qué es el MACD?"):
        st.markdown("""
        ### 📊 MACD
        
        **Señales:**
        - **MACD cruza arriba de Señal:** Compra
        - **MACD cruza abajo de Señal:** Venta
        - **Histograma positivo:** Momentum alcista
        - **Histograma negativo:** Momentum bajista
        """)
    
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
            name='Señal',
            line=dict(color='#FF6B6B', width=2)
        ))
        
        colors_hist = [COLORS['success'] if val >= 0 else COLORS['danger'] for val in df['macd_hist']]
        
        fig_macd.add_trace(go.Bar(
            x=df.index,
            y=df['macd_hist'],
            name='Histograma',
            marker_color=colors_hist,
            opacity=0.5
        ))
        
        fig_macd.add_hline(y=0, line_dash="solid", line_color="#666", line_width=1)
        
        fig_macd.update_layout(
            height=300,
            template='plotly_dark',
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_macd, use_container_width=True)
    
    st.markdown("---")
    
    # Resumen de indicadores
    st.markdown("### 📋 Resumen de Indicadores")
    
    indicators_data = []
    
    if 'rsi' in df.columns:
        rsi_val = df['rsi'].iloc[-1]
        if rsi_val > 70:
            rsi_int = "🔴 Sobrecompra - Posible corrección"
        elif rsi_val < 30:
            rsi_int = "🟢 Sobreventa - Posible rebote"
        else:
            rsi_int = "🟡 Neutral"
        
        indicators_data.append({
            'Indicador': 'RSI (14)',
            'Valor': f'{rsi_val:.2f}',
            'Interpretación': rsi_int
        })
    
    if 'macd' in df.columns:
        macd_val = df['macd'].iloc[-1]
        macd_sig = df['macd_signal'].iloc[-1]
        if macd_val > macd_sig:
            macd_int = "🟢 Alcista - MACD > Señal"
        else:
            macd_int = "🔴 Bajista - MACD < Señal"
        
        indicators_data.append({
            'Indicador': 'MACD',
            'Valor': f'{macd_val:.4f}',
            'Interpretación': macd_int
        })
    
    if 'ema_50' in df.columns and 'ema_200' in df.columns:
        ema50 = df['ema_50'].iloc[-1]
        ema200 = df['ema_200'].iloc[-1]
        if ema50 > ema200:
            ema_int = "🟢 Tendencia Alcista"
        else:
            ema_int = "🔴 Tendencia Bajista"
        
        indicators_data.append({
            'Indicador': 'EMAs (50/200)',
            'Valor': f'50: ${ema50:.2f} | 200: ${ema200:.2f}',
            'Interpretación': ema_int
        })
    
    df_indicators = pd.DataFrame(indicators_data)
    st.dataframe(df_indicators, use_container_width=True, hide_index=True)

# TAB 4: ALERTAS
with tab4:
    st.markdown("## 🔔 Sistema de Alertas")
    st.caption("Configura notificaciones personalizadas")
    
    st.info("🚧 Notificaciones por Telegram/Email disponibles en versión PRO")
    
    with st.expander("ℹ️ ¿Cómo funcionan las alertas?"):
        st.markdown("""
        ### 🔔 Sistema de Alertas
        
        **Tipos disponibles:**
        - **Precio:** Alerta cuando alcanza un valor
        - **RSI:** Alerta en sobrecompra/sobreventa
        - **Cambio %:** Alerta en movimientos bruscos
        
        **En versión PRO:**
        - 🤖 Alertas de predicciones ML
        - 📱 Notificaciones Telegram instantáneas
        - 📧 Alertas por email
        - 🎯 Alertas de cruces de indicadores
        """)
    
    with st.form("alert_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            alert_type = st.selectbox(
                "📌 Tipo de Alerta",
                ["Precio", "RSI", "Cambio %"]
            )
            condition = st.selectbox(
                "🎯 Condición",
                ["Mayor que", "Menor que"]
            )
        
        with col2:
            if alert_type == "Precio":
                threshold = st.number_input(
                    "💵 Valor del Precio (USDT)",
                    value=float(current_price),
                    step=100.0
                )
            else:
                threshold = st.number_input(
                    "📊 Valor Umbral",
                    value=70.0,
                    step=5.0
                )
        
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
            st.success(f"✅ Alerta creada: {crypto_symbol} - {alert_type} {condition} {threshold}")
    
    if st.session_state.alerts:
        st.markdown("### 📋 Alertas Activas")
        for i, alert in enumerate(st.session_state.alerts):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{alert['crypto']}** - {alert['type']} {alert['condition']} {alert['threshold']}")
                st.caption(f"Creada: {alert['created'].strftime('%Y-%m-%d %H:%M')}")
            with col2:
                if st.button("🗑️ Eliminar", key=f"del_{i}"):
                    st.session_state.alerts.pop(i)
                    st.rerun()
    else:
        st.info("📭 No tienes alertas configuradas")

# ============ FOOTER ============
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.caption("🚀 CryptoView Pro v1.0")
with col_f2:
    st.caption("👨‍💻 Developed by **Julian E. Coronado Gil** - Data Scientist")
with col_f3:
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Disclaimer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 12px; padding: 20px;'>
    <p>⚠️ <strong>AVISO IMPORTANTE:</strong> Este sistema proporciona análisis técnico educativo.</p>
    <p>NO es asesoría financiera. Las criptomonedas son volátiles. Invierte responsablemente.</p>
</div>
""", unsafe_allow_html=True)
