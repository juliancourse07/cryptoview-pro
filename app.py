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
    st.caption("🎯 Horas hacia el futuro para predecir (disponible en versión PRO)")
    forecast_hours = st.slider(
        "Horas:",
        min_value=1,
        max_value=168,
        value=FORECAST_HOURS,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Configuración de Datos")
    
    # CANTIDAD DE DATOS
    st.caption("💾 Cantidad de puntos históricos. Más datos = análisis más completo pero más lento")
    data_limit = st.number_input(
        "Puntos de datos:",
        min_value=100,
        max_value=2000,
        value=DATA_LIMIT,
        step=100,
        label_visibility="collapsed"
    )
    
    # MOSTRAR VOLUMEN
    show_volume = st.checkbox("📈 Mostrar volumen de transacciones", value=True)
    st.caption("💡 El volumen indica cuánto dinero se mueve. Alto volumen = movimiento significativo")
    
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
    """Crea gráfico principal"""
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
                name='EMA 50 (Media Móvil Rápida)',
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
                name='EMA 200 (Tendencia General)',
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
        title=f"{crypto_symbol} - Análisis en Tiempo Real",
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

# Display métricas con tooltips
st.markdown("### 📊 Métricas Principales")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 14px; color: #9CA3AF;'>💰 Precio Actual</div>
        <div class='big-metric'>${current_price:,.2f}</div>
        <div style='font-size: 11px; color: #6B7280; margin-top: 4px;'>
            Último precio registrado en el mercado
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
            Variación en las últimas 24 horas
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
            Total transaccionado en 24h. Alto volumen = alta actividad
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    rsi_color = COLORS['danger'] if current_rsi > 70 else COLORS['success'] if current_rsi < 30 else COLORS['warning']
    rsi_status = "Sobrecompra" if current_rsi > 70 else "Sobreventa" if current_rsi < 30 else "Neutral"
    rsi_explanation = "Posible caída" if current_rsi > 70 else "Posible subida" if current_rsi < 30 else "Sin extremos"
    
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 14px; color: #9CA3AF;'>🎯 RSI (Fuerza Relativa)</div>
        <div style='font-size: 24px; font-weight: 700; color: {rsi_color};'>
            {current_rsi:.1f}
        </div>
        <div style='font-size: 12px; color: #6B7280;'>{rsi_status}</div>
        <div style='font-size: 11px; color: #6B7280; margin-top: 4px;'>
            {rsi_explanation}. Rango: 0-100
        </div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    signal_description = {
        'buy': 'Los indicadores sugieren oportunidad de compra',
        'sell': 'Los indicadores sugieren considerar venta',
        'neutral': 'Sin señales claras, esperar'
    }.get(overall_signal, 'Analizando...')
    
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 14px; color: #9CA3AF;'>🎲 Señal de Trading</div>
        <div style='font-size: 28px; font-weight: 700;'>{signal_emoji}</div>
        <div style='font-size: 14px; color: #6B7280; font-weight: 600; text-transform: uppercase;'>{signal_text}</div>
        <div style='font-size: 11px; color: #6B7280; margin-top: 4px;'>
            {signal_description}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Explicación general de métricas
with st.expander("ℹ️ ¿Qué significan estas métricas?"):
    st.markdown("""
    ### 📚 Guía de Indicadores
    
    **💰 Precio Actual**
    - Es el último precio al que se compró/vendió la criptomoneda
    - Se actualiza en tiempo real según el exchange
    
    **📈 Cambio 24h**
    - Muestra cuánto subió o bajó en las últimas 24 horas
    - Verde (+) = subió | Rojo (-) = bajó
    - Ejemplo: +5% significa que si costaba $100, ahora vale $105
    
    **📊 Volumen 24h**
    - Total de dinero que se movió en 24 horas
    - Alto volumen con subida = tendencia fuerte alcista
    - Alto volumen con bajada = tendencia fuerte bajista
    - Bajo volumen = movimiento débil, puede revertirse
    
    **🎯 RSI (Índice de Fuerza Relativa)**
    - Mide si está "muy comprado" o "muy vendido"
    - RSI > 70: Sobrecompra (puede bajar pronto)
    - RSI < 30: Sobreventa (puede subir pronto)
    - RSI 40-60: Neutral, sin extremos
    
    **🎲 Señal de Trading**
    - Combina varios indicadores para dar una recomendación
    - 🟢 COMPRA: Condiciones favorables para entrar
    - 🔴 VENTA: Condiciones sugieren salir o no entrar
    - 🟡 NEUTRAL: Esperar mejores condiciones
    - ⚠️ **IMPORTANTE:** Esto NO es asesoría financiera, solo análisis técnico
    """)

st.markdown("---")

# ============ TABS ============
tab1, tab2, tab3 = st.tabs(["🏠 Panel Principal", "📊 Análisis Técnico Detallado", "🔔 Sistema de Alertas"])

# TAB 1: DASHBOARD
with tab1:
    st.markdown("### 📈 Gráfico de Velas (Candlestick)")
    
    # Explicación de gráfico
    with st.expander("📖 ¿Cómo leer el gráfico de velas?"):
        st.markdown("""
        ### 🕯️ Interpretación de Velas Japonesas
        
        **Vela Verde (Alcista):**
        - El precio SUBIÓ en ese período
        - La parte inferior = precio de apertura
        - La parte superior = precio de cierre
        - Las líneas (mechas) = máximo y mínimo alcanzados
        
        **Vela Roja (Bajista):**
        - El precio BAJÓ en ese período
        - La parte superior = precio de apertura
        - La parte inferior = precio de cierre
        
        **Líneas Naranjas y Moradas:**
        - EMA 50 (naranja): Media móvil de 50 períodos, muestra tendencia reciente
        - EMA 200 (morada): Media móvil de 200 períodos, muestra tendencia general
        - Si EMA 50 está ARRIBA de EMA 200 = tendencia alcista
        - Si EMA 50 está ABAJO de EMA 200 = tendencia bajista
        
        **Gráfico de Volumen (abajo):**
        - Muestra cuánto se comerció en cada período
        - Barras verdes = compradores dominaron
        - Barras rojas = vendedores dominaron
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
    st.caption("Cada fila representa un período de tiempo según el marco temporal seleccionado")
    recent_data = df[['open', 'high', 'low', 'close', 'volume']].tail(10)
    recent_data.columns = ['Apertura', 'Máximo', 'Mínimo', 'Cierre', 'Volumen']
    st.dataframe(recent_data, use_container_width=True)

# TAB 2: ANÁLISIS TÉCNICO
with tab2:
    st.markdown("## 📊 Análisis Técnico Detallado")
    st.caption("Indicadores avanzados para tomar decisiones de trading más informadas")
    
    # RSI
    st.markdown("### 📉 RSI - Índice de Fuerza Relativa")
    
    with st.expander("ℹ️ ¿Qué es el RSI y cómo usarlo?"):
        st.markdown("""
        ### 🎯 RSI (Relative Strength Index)
        
        **¿Qué mide?**
        - La velocidad y magnitud de los cambios de precio
        - Identifica si un activo está sobrecomprado o sobrevendido
        
        **Cómo interpretarlo:**
        - **RSI > 70:** Zona de SOBRECOMPRA
          - El precio subió muy rápido
          - Posible corrección a la baja
          - Considera esperar antes de comprar
        
        - **RSI < 30:** Zona de SOBREVENTA
          - El precio bajó muy rápido
          - Posible rebote al alza
          - Puede ser oportunidad de compra
        
        - **RSI 40-60:** Zona NEUTRAL
          - Sin señales extremas
          - Mercado en equilibrio
        
        **💡 Tip:** Usa RSI junto con otros indicadores, no solo.
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
            yaxis_title="RSI"
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    st.markdown("---")
    
    # MACD
    st.markdown("### 📈 MACD - Convergencia/Divergencia de Medias Móviles")
    
    with st.expander("ℹ️ ¿Qué es el MACD y cómo usarlo?"):
        st.markdown("""
        ### 📊 MACD (Moving Average Convergence Divergence)
        
        **¿Qué mide?**
        - La relación entre dos medias móviles
        - Identifica cambios en la tendencia
        
        **Componentes:**
        - **Línea MACD (azul):** Diferencia entre medias rápida y lenta
        - **Línea de Señal (roja):** Media de la línea MACD
        - **Histograma (barras):** Diferencia entre MACD y Señal
        
        **Señales de Trading:**
        
        **🟢 SEÑAL DE COMPRA:**
        - Línea MACD cruza ARRIBA de la línea de Señal
        - Histograma pasa de negativo a positivo
        - Sugiere inicio de tendencia alcista
        
        **🔴 SEÑAL DE VENTA:**
        - Línea MACD cruza ABAJO de la línea de Señal
        - Histograma pasa de positivo a negativo
        - Sugiere inicio de tendencia bajista
        
        **💡 Tip:** Los cruces son más confiables cuando ocurren lejos del cero.
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
            showlegend=True
        )
        
        st.plotly_chart(fig_macd, use_container_width=True)
    
    st.markdown("---")
    
    # Tabla de indicadores
    st.markdown("### 📋 Resumen de Indicadores Actuales")
    st.caption("Estado actual de todos los indicadores técnicos")
    
    indicators_data = []
    
    if 'rsi' in df.columns:
        rsi_val = df['rsi'].iloc[-1]
        rsi_interpretation = ""
        if rsi_val > 70:
            rsi_interpretation = "🔴 Sobrecompra - Posible corrección a la baja"
        elif rsi_val < 30:
            rsi_interpretation = "🟢 Sobreventa - Posible rebote al alza"
        else:
            rsi_interpretation = "🟡 Neutral - Sin señales extremas"
            
        indicators_data.append({
            'Indicador': 'RSI (14)',
            'Valor Actual': f'{rsi_val:.2f}',
            'Interpretación': rsi_interpretation
        })
    
    if 'macd' in df.columns:
        macd_val = df['macd'].iloc[-1]
        macd_sig = df['macd_signal'].iloc[-1]
        macd_interpretation = ""
        if macd_val > macd_sig:
            macd_interpretation = "🟢 Alcista - MACD por encima de señal"
        else:
            macd_interpretation = "🔴 Bajista - MACD por debajo de señal"
            
        indicators_data.append({
            'Indicador': 'MACD',
            'Valor Actual': f'{macd_val:.4f}',
            'Interpretación': macd_interpretation
        })
    
    # EMAs
    if 'ema_50' in df.columns and 'ema_200' in df.columns:
        ema50 = df['ema_50'].iloc[-1]
        ema200 = df['ema_200'].iloc[-1]
        ema_interpretation = ""
        if ema50 > ema200:
            ema_interpretation = "🟢 Tendencia Alcista - EMA50 > EMA200"
        else:
            ema_interpretation = "🔴 Tendencia Bajista - EMA50 < EMA200"
            
        indicators_data.append({
            'Indicador': 'Cruces de EMAs',
            'Valor Actual': f'50: ${ema50:.2f} | 200: ${ema200:.2f}',
            'Interpretación': ema_interpretation
        })
    
    df_indicators = pd.DataFrame(indicators_data)
    st.dataframe(df_indicators, use_container_width=True, hide_index=True)

# TAB 3: ALERTAS
with tab3:
    st.markdown("## 🔔 Sistema de Alertas")
    st.caption("Configura notificaciones para no perderte movimientos importantes")
    
    st.info("🚧 Sistema de alertas básico. Notificaciones por email y Telegram disponibles en versión PRO.")
    
    with st.expander("ℹ️ ¿Cómo funcionan las alertas?"):
        st.markdown("""
        ### 🔔 Sistema de Alertas
        
        **¿Para qué sirven?**
        - Te notifican cuando se cumplen condiciones específicas
        - No necesitas estar mirando el gráfico todo el tiempo
        - Útil para entrar/salir en momentos clave
        
        **Tipos de alertas disponibles:**
        
        **💰 Alerta de Precio:**
        - Te avisa cuando el precio llega a un valor específico
        - Ejemplo: "Avísame cuando BTC llegue a $45,000"
        
        **🎯 Alerta de RSI:**
        - Te avisa cuando RSI entra en zona de sobrecompra/sobreventa
        - Ejemplo: "Avísame cuando RSI baje de 30 (sobreventa)"
        
        **📈 Alerta de Cambio %:**
        - Te avisa cuando hay movimientos bruscos
        - Ejemplo: "Avísame si sube o baja más de 5% en 24h"
        
        **💡 Tip:** Configura varias alertas para diferentes escenarios.
        """)
    
    with st.form("alert_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            alert_type = st.selectbox(
                "📌 Tipo de Alerta",
                ["Precio", "RSI", "Cambio %"],
                help="Elige qué tipo de condición quieres monitorear"
            )
            condition = st.selectbox(
                "🎯 Condición",
                ["Mayor que", "Menor que"],
                help="Define cuándo debe activarse la alerta"
            )
        
        with col2:
            if alert_type == "Precio":
                threshold = st.number_input(
                    "💵 Valor del Precio (USDT)",
                    value=float(current_price),
                    step=100.0,
                    help="Precio al que quieres ser notificado"
                )
            else:
                threshold = st.number_input(
                    "📊 Valor Umbral",
                    value=70.0,
                    step=5.0,
                    help="Valor de RSI o porcentaje de cambio"
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
            st.info("💡 En versión PRO recibirás notificaciones por email/Telegram")
    
    if st.session_state.alerts:
        st.markdown("### 📋 Alertas Activas")
        st.caption("Tus alertas configuradas actualmente")
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
        st.info("📭 No tienes alertas configuradas. ¡Crea tu primera alerta arriba!")

# ============ FOOTER ============
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.caption("🚀 CryptoView Pro v1.0")
with col_f2:
    st.caption("👨‍💻 Developed by **Julian E. Coronado Gil** - Data Scientist")
with col_f3:
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Info adicional al final
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 12px; padding: 20px;'>
    <p>⚠️ <strong>AVISO IMPORTANTE:</strong> Este sistema proporciona análisis técnico educativo.</p>
    <p>NO es asesoría financiera. Siempre investiga y consulta profesionales antes de invertir.</p>
    <p>Las criptomonedas son volátiles y puedes perder tu inversión.</p>
</div>
""", unsafe_allow_html=True)
