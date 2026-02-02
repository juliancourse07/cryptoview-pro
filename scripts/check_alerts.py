"""
🔔 Sistema de Alertas Inteligentes - CryptoView Pro
Top 10 Cryptos - Análisis Semanal - Alertas Avanzadas Nivel 2

Developed by Julian E. Coronado Gil - Data Scientist
"""
import ccxt
import requests
import os
import json
from datetime import datetime
from pathlib import Path

# ============ CONFIGURACIÓN ============
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Top 10 cryptos por capitalización
TOP_CRYPTOS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT'
]

# Niveles psicológicos
PSYCHOLOGICAL_LEVELS = {
    'BTC/USDT': [100000, 90000, 80000, 75000, 70000, 60000, 50000],
    'ETH/USDT': [5000, 4000, 3500, 3000, 2500, 2000],
    'BNB/USDT': [700, 600, 500, 400, 300],
    'SOL/USDT': [200, 150, 100, 75, 50],
    'XRP/USDT': [3, 2.5, 2, 1.5, 1],
}

ALERTS_FILE = Path(__file__).parent / 'alerts_config.json'

# ============ TELEGRAM ============

def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    
    try:
        response = requests.post(url, data=data, timeout=10)
        return response.status_code == 200
    except:
        return False

# ============ ANÁLISIS ============

def get_price_data(symbol: str, exchange_name: str = 'kraken'):
    """Obtiene datos de precio y volumen"""
    try:
        exchange = ccxt.kraken() if exchange_name == 'kraken' else ccxt.binance()
        ticker = exchange.fetch_ticker(symbol)
        ohlcv_1d = exchange.fetch_ohlcv(symbol, '1d', limit=7)
        ohlcv_1h = exchange.fetch_ohlcv(symbol, '1h', limit=200)
        
        return {
            'current': ticker['last'],
            'volume_24h': ticker['quoteVolume'],
            'ohlcv_7d': ohlcv_1d,
            'ohlcv_200h': ohlcv_1h
        }
    except Exception as e:
        print(f"❌ Error {symbol}: {e}")
        return None


def analyze_weekly_range(symbol: str, data: dict) -> dict:
    """Analiza mínimo y máximo de la última semana"""
    try:
        ohlcv = data['ohlcv_7d']
        
        lows = [c[3] for c in ohlcv]
        highs = [c[2] for c in ohlcv]
        
        weekly_low = min(lows)
        weekly_high = max(highs)
        current = data['current']
        
        # Calcular cambio semanal
        price_7d_ago = ohlcv[0][4]
        weekly_change = ((current - price_7d_ago) / price_7d_ago) * 100
        
        return {
            'low': weekly_low,
            'high': weekly_high,
            'current': current,
            'weekly_change': weekly_change,
            'support_broken': current < weekly_low,
            'resistance_broken': current > weekly_high,
            'distance_from_low': ((current - weekly_low) / weekly_low) * 100,
            'distance_from_high': ((current - weekly_high) / weekly_high) * 100
        }
    except Exception as e:
        print(f"❌ Error análisis semanal: {e}")
        return None


def calculate_ema(prices: list, period: int) -> float:
    """Calcula EMA"""
    try:
        k = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * k) + (ema * (1 - k))
        return ema
    except:
        return None


def detect_ema_cross(data: dict) -> dict:
    """Detecta Golden Cross (50/200) o Death Cross"""
    try:
        closes = [c[4] for c in data['ohlcv_200h']]
        
        if len(closes) < 200:
            return {'cross': 'insufficient_data'}
        
        ema_50_current = calculate_ema(closes[-50:], 50)
        ema_200_current = calculate_ema(closes[-200:], 200)
        
        ema_50_prev = calculate_ema(closes[-51:-1], 50)
        ema_200_prev = calculate_ema(closes[-201:-1], 200)
        
        # Detectar cruce
        if ema_50_prev < ema_200_prev and ema_50_current > ema_200_current:
            return {'cross': 'golden', 'ema_50': ema_50_current, 'ema_200': ema_200_current}
        elif ema_50_prev > ema_200_prev and ema_50_current < ema_200_current:
            return {'cross': 'death', 'ema_50': ema_50_current, 'ema_200': ema_200_current}
        else:
            return {
                'cross': 'none',
                'ema_50': ema_50_current,
                'ema_200': ema_200_current,
                'position': 'bullish' if ema_50_current > ema_200_current else 'bearish'
            }
    except Exception as e:
        print(f"❌ Error EMA: {e}")
        return {'cross': 'error'}


def calculate_rsi(data: dict, period: int = 14) -> float:
    """Calcula RSI"""
    try:
        closes = [c[4] for c in data['ohlcv_200h'][-period-1:]]
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except:
        return None


def detect_volume_spike(data: dict) -> dict:
    """Detecta volumen anormal >200%"""
    try:
        volumes = [c[5] for c in data['ohlcv_7d'][:-1]]  # Últimos 6 días
        avg_volume = sum(volumes) / len(volumes)
        current_volume = data['volume_24h']
        
        spike_ratio = (current_volume / avg_volume) if avg_volume > 0 else 1
        
        return {
            'current': current_volume,
            'average': avg_volume,
            'spike_ratio': spike_ratio,
            'is_spike': spike_ratio > 2.0
        }
    except:
        return {'is_spike': False}


def check_psychological_level(symbol: str, price: float) -> dict:
    """Verifica si está cerca de nivel psicológico"""
    if symbol not in PSYCHOLOGICAL_LEVELS:
        return {'near_level': False}
    
    levels = PSYCHOLOGICAL_LEVELS[symbol]
    
    for level in levels:
        distance_pct = abs((price - level) / level) * 100
        
        # Si está a menos de 1% del nivel
        if distance_pct < 1.0:
            return {
                'near_level': True,
                'level': level,
                'distance_pct': distance_pct,
                'above': price > level
            }
    
    return {'near_level': False}


# ============ ALERTAS ============

def send_support_break_alert(symbol: str, analysis: dict):
    """Alerta: Mínimo semanal roto - OPORTUNIDAD DE COMPRA"""
    message = f"""
🟢 *OPORTUNIDAD DE COMPRA - {symbol}*

💎 *Precio rompió MÍNIMO SEMANAL*

📉 *Mínimo 7 días:* ${analysis['low']:,.2f}
📊 *Precio Actual:* ${analysis['current']:,.2f}
💚 *Caída:* {analysis['distance_from_low']:.2f}%

📈 *Cambio Semanal:* {analysis['weekly_change']:+.2f}%

✨ *Señal Técnica:*
• Soporte semanal quebrado
• Posible reversión alcista
• Zona de acumulación

🎯 *Estrategia de Entrada:*
• Entrada escalonada (3 partes)
• Stop-loss: {analysis['low'] * 0.97:.2f} (-3%)
• Target 1: {analysis['low'] * 1.05:.2f} (+5%)
• Target 2: {analysis['high']:.2f} (máximo semanal)

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

_CryptoView Pro by Julian E. Coronado Gil_
"""
    send_telegram(message)


def send_resistance_break_alert(symbol: str, analysis: dict):
    """Alerta: Máximo semanal roto - BREAKOUT"""
    message = f"""
🚀 *BREAKOUT - {symbol}*

💰 *Precio rompió MÁXIMO SEMANAL*

📈 *Máximo 7 días:* ${analysis['high']:,.2f}
📊 *Precio Actual:* ${analysis['current']:,.2f}
💚 *Ganancia:* {analysis['distance_from_high']:+.2f}%

📈 *Cambio Semanal:* {analysis['weekly_change']:+.2f}%

✨ *Señal Técnica:*
• Resistencia semanal rota
• Momentum alcista fuerte
• Posible continuación

🎯 *Estrategia:*
• Esperar pullback a {analysis['high']:.2f}
• Trailing stop recomendado
• Tomar parciales en extensiones

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

_CryptoView Pro by Julian E. Coronado Gil_
"""
    send_telegram(message)


def send_golden_cross_alert(symbol: str, ema_data: dict):
    """Alerta: Golden Cross detectado"""
    message = f"""
🌟 *GOLDEN CROSS - {symbol}*

📈 *EMA 50 cruzó ARRIBA de EMA 200*

📊 *Indicadores:*
• EMA 50: ${ema_data['ema_50']:,.2f}
• EMA 200: ${ema_data['ema_200']:,.2f}

✨ *Señal Técnica:*
• Cruce alcista confirmado
• Tendencia de largo plazo positiva
• Alta probabilidad de rally

🎯 *Estrategia:*
• Entrada en pullback
• Hold de mediano plazo
• Stop bajo EMA 200

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

_CryptoView Pro by Julian E. Coronado Gil_
"""
    send_telegram(message)


def send_death_cross_alert(symbol: str, ema_data: dict):
    """Alerta: Death Cross detectado"""
    message = f"""
⚠️ *DEATH CROSS - {symbol}*

📉 *EMA 50 cruzó ABAJO de EMA 200*

📊 *Indicadores:*
• EMA 50: ${ema_data['ema_50']:,.2f}
• EMA 200: ${ema_data['ema_200']:,.2f}

⚠️ *Señal Técnica:*
• Cruce bajista confirmado
• Tendencia de largo plazo negativa
• Precaución recomendada

🛡️ *Estrategia:*
• Reducir exposición
• Stop-loss ajustados
• Esperar confirmación de reversión

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

_CryptoView Pro by Julian E. Coronado Gil_
"""
    send_telegram(message)


def send_volume_spike_alert(symbol: str, vol_data: dict, price: float):
    """Alerta: Volumen anormal"""
    message = f"""
📊 *VOLUMEN ANORMAL - {symbol}*

💥 *Volumen {vol_data['spike_ratio']:.1f}x el promedio*

📈 *Precio Actual:* ${price:,.2f}
📊 *Volumen 24h:* ${vol_data['current']:,.0f}
📉 *Promedio 7d:* ${vol_data['average']:,.0f}

⚠️ *Implicaciones:*
• Interés institucional aumentado
• Posible movimiento fuerte próximo
• Revisar noticias y contexto

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

_CryptoView Pro by Julian E. Coronado Gil_
"""
    send_telegram(message)


def send_rsi_extreme_alert(symbol: str, rsi: float, price: float):
    """Alerta: RSI extremo"""
    if rsi < 25:
        condition = "SOBREVENTA EXTREMA"
        emoji = "🟢"
        signal = "Posible rebote alcista"
    else:
        condition = "SOBRECOMPRA EXTREMA"
        emoji = "🔴"
        signal = "Posible corrección bajista"
    
    message = f"""
{emoji} *RSI EXTREMO - {symbol}*

📊 *{condition}*

📈 *Precio:* ${price:,.2f}
📉 *RSI:* {rsi:.1f}

⚠️ *Señal:* {signal}

🎯 *Estrategia:*
{'• Zona de acumulación' if rsi < 25 else '• Considerar tomar ganancias'}
{'• Esperar confirmación' if rsi < 25 else '• Ajustar stop-loss'}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

_CryptoView Pro by Julian E. Coronado Gil_
"""
    send_telegram(message)


def send_psychological_level_alert(symbol: str, level_data: dict):
    """Alerta: Cerca de nivel psicológico"""
    direction = "ARRIBA" if level_data['above'] else "ABAJO"
    message = f"""
🎯 *NIVEL PSICOLÓGICO - {symbol}*

💰 *Precio cerca de ${level_data['level']:,.0f}*

📊 *Posición:* {direction}
📏 *Distancia:* {level_data['distance_pct']:.2f}%

⚠️ *Zona de alta reacción:*
• Posible rebote o ruptura
• Alto volumen esperado
• Monitorear de cerca

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

_CryptoView Pro by Julian E. Coronado Gil_
"""
    send_telegram(message)


# ============ ANÁLISIS COMPLETO ============

def analyze_crypto(symbol: str, exchange: str = 'kraken') -> dict:
    """Análisis completo de una crypto"""
    print(f"\n{'='*50}")
    print(f"Analizando: {symbol}")
    print(f"{'='*50}")
    
    # Obtener datos
    data = get_price_data(symbol, exchange)
    if not data:
        return None
    
    # Análisis
    weekly = analyze_weekly_range(symbol, data)
    ema_cross = detect_ema_cross(data)
    rsi = calculate_rsi(data)
    volume = detect_volume_spike(data)
    psych_level = check_psychological_level(symbol, data['current'])
    
    alerts_triggered = []
    
    # NIVEL 1: Soporte/Resistencia Semanal
    if weekly['support_broken']:
        print(f"🟢 SOPORTE ROTO - Oportunidad de compra")
        send_support_break_alert(symbol, weekly)
        alerts_triggered.append('support_broken')
    
    if weekly['resistance_broken']:
        print(f"🚀 RESISTENCIA ROTA - Breakout")
        send_resistance_break_alert(symbol, weekly)
        alerts_triggered.append('resistance_broken')
    
    # NIVEL 2: Golden/Death Cross
    if ema_cross['cross'] == 'golden':
        print(f"🌟 GOLDEN CROSS")
        send_golden_cross_alert(symbol, ema_cross)
        alerts_triggered.append('golden_cross')
    elif ema_cross['cross'] == 'death':
        print(f"⚠️ DEATH CROSS")
        send_death_cross_alert(symbol, ema_cross)
        alerts_triggered.append('death_cross')
    
    # NIVEL 2: Volumen anormal
    if volume['is_spike']:
        print(f"📊 VOLUMEN ANORMAL: {volume['spike_ratio']:.1f}x")
        send_volume_spike_alert(symbol, volume, data['current'])
        alerts_triggered.append('volume_spike')
    
    # NIVEL 2: RSI extremo
    if rsi and (rsi < 25 or rsi > 75):
        print(f"⚠️ RSI EXTREMO: {rsi:.1f}")
        send_rsi_extreme_alert(symbol, rsi, data['current'])
        alerts_triggered.append('rsi_extreme')
    
    # NIVEL 2: Nivel psicológico
    if psych_level['near_level']:
        print(f"🎯 CERCA DE NIVEL: ${psych_level['level']:,.0f}")
        send_psychological_level_alert(symbol, psych_level)
        alerts_triggered.append('psychological_level')
    
    if not alerts_triggered:
        print(f"✓ Sin alertas")
    
    return {
        'symbol': symbol,
        'price': data['current'],
        'weekly_change': weekly['weekly_change'],
        'rsi': rsi,
        'alerts': alerts_triggered
    }


def generate_weekly_report(results: list):
    """Genera reporte semanal de las top 10"""
    # Ordenar por ganancia semanal
    sorted_results = sorted(results, key=lambda x: x['weekly_change'], reverse=True)
    
    report = "📊 *REPORTE SEMANAL - TOP 10 CRYPTOS*\n\n"
    
    report += "🏆 *Ranking por Ganancia Semanal:*\n\n"
    
    for i, r in enumerate(sorted_results, 1):
        emoji = "🟢" if r['weekly_change'] > 0 else "🔴"
        rsi_status = ""
        if r['rsi']:
            if r['rsi'] < 30:
                rsi_status = " 🟢RSI:Bajo"
            elif r['rsi'] > 70:
                rsi_status = " 🔴RSI:Alto"
        
        alerts_str = f" 🔔{len(r['alerts'])}" if r['alerts'] else ""
        
        report += f"{i}. {emoji} *{r['symbol'].split('/')[0]}*\n"
        report += f"   ${r['price']:,.2f} | {r['weekly_change']:+.2f}%{rsi_status}{alerts_str}\n\n"
    
    report += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += "\n_CryptoView Pro by Julian E. Coronado Gil_"
    
    send_telegram(report)


# ============ MAIN ============

def main():
    print("="*60)
    print("🔔 CRYPTOVIEW PRO - ALERTAS INTELIGENTES")
    print("   Top 10 Cryptos + Análisis Semanal + Nivel 2")
    print("="*60)
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👨‍💻 Julian E. Coronado Gil")
    print("="*60)
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n❌ ERROR: Secrets no configurados")
        return
    
    results = []
    total_alerts = 0
    
    # Analizar cada crypto
    for crypto in TOP_CRYPTOS:
        result = analyze_crypto(crypto)
        if result:
            results.append(result)
            total_alerts += len(result['alerts'])
    
    # Generar reporte semanal
    if results:
        generate_weekly_report(results)
    
    print("\n" + "="*60)
    print(f"✅ ANÁLISIS COMPLETADO")
    print(f"📊 Cryptos analizadas: {len(results)}/10")
    print(f"🔔 Alertas activadas: {total_alerts}")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            send_telegram(f"🚨 *ERROR Sistema Alertas*\n\n```\n{str(e)}\n```")
