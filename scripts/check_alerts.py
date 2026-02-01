"""
🔔 Sistema Automático de Alertas Inteligentes para CryptoView Pro
Ejecutado por GitHub Actions cada hora

Detecta:
- Rupturas de soporte (mínimo mensual quebrado)
- Rupturas de resistencia (máximo mensual superado)
- RSI extremo
- Cambios bruscos en 24h

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

# Cargar alertas desde archivo JSON
ALERTS_FILE = Path(__file__).parent / 'alerts_config.json'

# ============ FUNCIONES DE TELEGRAM ============

def send_telegram(message: str, parse_mode: str = 'Markdown') -> bool:
    """
    Envía mensaje a Telegram
    
    Args:
        message: Texto del mensaje
        parse_mode: 'Markdown' o 'HTML'
        
    Returns:
        True si se envió exitosamente
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Error: TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": parse_mode
    }
    
    try:
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            print(f"✅ Mensaje enviado a Telegram")
            return True
        else:
            print(f"❌ Error al enviar: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False


def send_alert_notification(alert: dict, current_price: float) -> bool:
    """
    Envía notificación formateada de alerta básica (precio, RSI)
    
    Args:
        alert: Diccionario con configuración de alerta
        current_price: Precio actual
        
    Returns:
        True si se envió
    """
    condition_text = {
        'mayor_que': 'Mayor que',
        'menor_que': 'Menor que',
        'igual_a': 'Igual a'
    }.get(alert['condition'], alert['condition'])
    
    emoji = "🔴" if alert['condition'] == 'menor_que' else "🟢"
    
    message = f"""
{emoji} *ALERTA AUTOMÁTICA - CRYPTOVIEW PRO*

💰 *{alert['crypto']}*
📊 Tipo: {alert['type']}

🎯 *Condición Configurada:*
{condition_text} ${alert['threshold']:,.2f}

📈 *Precio Actual:*
${current_price:,.2f}

💡 *Diferencia:*
${abs(current_price - alert['threshold']):,.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

_Alerta automática ejecutada por GitHub Actions_
_CryptoView Pro by Julian E. Coronado Gil_
"""
    
    return send_telegram(message)


def send_support_break_alert(crypto: str, monthly_data: dict) -> bool:
    """
    Envía alerta cuando se rompe el soporte (mínimo mensual)
    
    Args:
        crypto: Símbolo
        monthly_data: Datos del mínimo mensual
        
    Returns:
        True si envió
    """
    message = f"""
🔴 *SOPORTE ROTO - ALERTA CRÍTICA*

💰 *{crypto}*

⚠️ *El precio ha caído por debajo del mínimo mensual*

📉 *Mínimo del último mes:*
${monthly_data['low']:,.2f}
📅 Fecha: {monthly_data['date'].strftime('%Y-%m-%d')}

📊 *Precio Actual:*
${monthly_data['current']:,.2f}

💔 *Caída desde mínimo:*
{monthly_data['pct_from_low']:.2f}%

⚠️ *Implicaciones:*
• Soporte técnico quebrado
• Posible tendencia bajista
• Alto riesgo de más caídas

🛡️ *Estrategia sugerida:*
• Stop-loss si estás en largo
• Esperar confirmación de rebote
• Considerar entradas escalonadas

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

_Alerta automática - CryptoView Pro_
_by Julian E. Coronado Gil_
"""
    
    return send_telegram(message)


def send_resistance_break_alert(crypto: str, monthly_data: dict) -> bool:
    """
    Envía alerta cuando se rompe la resistencia (máximo mensual)
    
    Args:
        crypto: Símbolo
        monthly_data: Datos del máximo mensual
        
    Returns:
        True si envió
    """
    message = f"""
🟢 *RESISTENCIA ROTA - BREAKOUT*

💰 *{crypto}*

🚀 *El precio ha superado el máximo mensual*

📈 *Máximo del último mes:*
${monthly_data['high']:,.2f}
📅 Fecha: {monthly_data['date'].strftime('%Y-%m-%d')}

📊 *Precio Actual:*
${monthly_data['current']:,.2f}

💚 *Ganancia desde máximo:*
{monthly_data['pct_from_high']:+.2f}%

✨ *Implicaciones:*
• Resistencia técnica quebrada
• Posible tendencia alcista fuerte
• Momentum positivo

🎯 *Estrategia sugerida:*
• Posible entrada en pullback
• Trailing stop recomendado
• Tomar parciales en niveles clave

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

_Alerta automática - CryptoView Pro_
_by Julian E. Coronado Gil_
"""
    
    return send_telegram(message)


# ============ FUNCIONES DE PRECIO ============

def get_current_price(symbol: str, exchange_name: str = 'kraken') -> float:
    """
    Obtiene precio actual de una criptomoneda
    
    Args:
        symbol: Símbolo del par (ej: 'BTC/USDT')
        exchange_name: Nombre del exchange ('kraken', 'binance')
        
    Returns:
        Precio actual o None si hay error
    """
    try:
        # Seleccionar exchange
        if exchange_name.lower() == 'kraken':
            exchange = ccxt.kraken()
        elif exchange_name.lower() == 'binance':
            exchange = ccxt.binance()
        else:
            exchange = ccxt.kraken()
        
        # Obtener ticker
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']
        
        print(f"📊 {symbol}: ${price:,.2f}")
        return price
        
    except Exception as e:
        print(f"❌ Error obteniendo precio de {symbol}: {e}")
        return None


def get_monthly_low(symbol: str, exchange_name: str = 'kraken') -> dict:
    """
    Obtiene el mínimo de los últimos 30 días
    
    Args:
        symbol: Par de crypto (ej: 'BTC/USDT')
        exchange_name: Exchange a usar
        
    Returns:
        Dict con {low, date, current_price, percentage_from_low, is_below}
    """
    try:
        if exchange_name.lower() == 'kraken':
            exchange = ccxt.kraken()
        elif exchange_name.lower() == 'binance':
            exchange = ccxt.binance()
        else:
            exchange = ccxt.kraken()
        
        # Obtener 30 días de datos (1 día por vela)
        ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=30)
        
        # Encontrar mínimo
        lows = [candle[3] for candle in ohlcv]  # index 3 = low
        dates = [candle[0] for candle in ohlcv]  # index 0 = timestamp
        
        min_price = min(lows)
        min_index = lows.index(min_price)
        min_date = datetime.fromtimestamp(dates[min_index] / 1000)
        
        # Precio actual
        current = get_current_price(symbol, exchange_name)
        
        # Calcular porcentaje desde el mínimo
        if current:
            pct_from_low = ((current - min_price) / min_price) * 100
        else:
            pct_from_low = 0
        
        result = {
            'low': min_price,
            'date': min_date,
            'current': current,
            'pct_from_low': pct_from_low,
            'is_below': current < min_price if current else False
        }
        
        print(f"📉 Mínimo 30d de {symbol}: ${min_price:,.2f} ({min_date.strftime('%Y-%m-%d')})")
        print(f"   Actual: ${current:,.2f} ({pct_from_low:+.2f}% desde mínimo)")
        
        return result
        
    except Exception as e:
        print(f"❌ Error obteniendo mínimo mensual de {symbol}: {e}")
        return None


def get_monthly_high(symbol: str, exchange_name: str = 'kraken') -> dict:
    """
    Obtiene el máximo de los últimos 30 días
    
    Args:
        symbol: Par de crypto
        exchange_name: Exchange
        
    Returns:
        Dict con información del máximo
    """
    try:
        if exchange_name.lower() == 'kraken':
            exchange = ccxt.kraken()
        else:
            exchange = ccxt.binance()
        
        ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=30)
        
        highs = [candle[2] for candle in ohlcv]  # index 2 = high
        dates = [candle[0] for candle in ohlcv]
        
        max_price = max(highs)
        max_index = highs.index(max_price)
        max_date = datetime.fromtimestamp(dates[max_index] / 1000)
        
        current = get_current_price(symbol, exchange_name)
        
        if current:
            pct_from_high = ((current - max_price) / max_price) * 100
        else:
            pct_from_high = 0
        
        result = {
            'high': max_price,
            'date': max_date,
            'current': current,
            'pct_from_high': pct_from_high,
            'is_above': current > max_price if current else False
        }
        
        print(f"📈 Máximo 30d de {symbol}: ${max_price:,.2f} ({max_date.strftime('%Y-%m-%d')})")
        print(f"   Actual: ${current:,.2f} ({pct_from_high:+.2f}% desde máximo)")
        
        return result
        
    except Exception as e:
        print(f"❌ Error obteniendo máximo mensual: {e}")
        return None


def calculate_rsi(symbol: str, period: int = 14) -> float:
    """
    Calcula RSI de una criptomoneda
    
    Args:
        symbol: Símbolo del par
        period: Período del RSI
        
    Returns:
        Valor RSI o None
    """
    try:
        exchange = ccxt.kraken()
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=period + 1)
        
        closes = [x[4] for x in ohlcv]
        
        # Calcular cambios
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        print(f"📈 RSI de {symbol}: {rsi:.2f}")
        return rsi
        
    except Exception as e:
        print(f"❌ Error calculando RSI: {e}")
        return None


def calculate_24h_change(symbol: str, exchange_name: str = 'kraken') -> dict:
    """
    Calcula el cambio porcentual en 24 horas
    
    Args:
        symbol: Par de crypto
        exchange_name: Exchange
        
    Returns:
        Dict con información del cambio
    """
    try:
        if exchange_name.lower() == 'kraken':
            exchange = ccxt.kraken()
        else:
            exchange = ccxt.binance()
        
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=25)
        
        price_24h_ago = ohlcv[-25][4]  # Cierre de hace 24h
        current = get_current_price(symbol, exchange_name)
        
        if current:
            change_pct = ((current - price_24h_ago) / price_24h_ago) * 100
            change_abs = current - price_24h_ago
        else:
            change_pct = 0
            change_abs = 0
        
        result = {
            'price_24h_ago': price_24h_ago,
            'current': current,
            'change_pct': change_pct,
            'change_abs': change_abs
        }
        
        print(f"📊 Cambio 24h de {symbol}: {change_pct:+.2f}%")
        
        return result
        
    except Exception as e:
        print(f"❌ Error calculando cambio 24h: {e}")
        return None


# ============ FUNCIONES DE ALERTAS ============

def load_alerts() -> list:
    """
    Carga alertas desde archivo JSON
    
    Returns:
        Lista de alertas
    """
    try:
        if ALERTS_FILE.exists():
            with open(ALERTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('alerts', [])
        else:
            print(f"⚠️ Archivo de alertas no encontrado: {ALERTS_FILE}")
            return []
    except Exception as e:
        print(f"❌ Error cargando alertas: {e}")
        return []


def check_alert(alert: dict) -> bool:
    """
    Verifica alertas básicas (precio, RSI)
    
    Args:
        alert: Diccionario con configuración de alerta
        
    Returns:
        True si se cumplió y envió notificación
    """
    if not alert.get('enabled', False):
        print(f"⏭️  Alerta deshabilitada: {alert.get('crypto')}")
        return False
    
    print(f"\n🔍 Revisando: {alert.get('crypto')} - {alert.get('type')}")
    
    # Obtener valor actual según tipo de alerta
    if alert['type'] == 'precio':
        current_value = get_current_price(alert['crypto'], alert.get('exchange', 'kraken'))
    elif alert['type'] == 'rsi':
        current_value = calculate_rsi(alert['crypto'])
    else:
        print(f"❌ Tipo de alerta no soportado: {alert['type']}")
        return False
    
    if current_value is None:
        print(f"❌ No se pudo obtener valor para {alert['crypto']}")
        return False
    
    # Verificar condición
    triggered = False
    condition = alert['condition']
    threshold = alert['threshold']
    
    if condition == 'mayor_que' and current_value > threshold:
        triggered = True
    elif condition == 'menor_que' and current_value < threshold:
        triggered = True
    elif condition == 'igual_a' and abs(current_value - threshold) < (threshold * 0.01):  # ±1%
        triggered = True
    
    if triggered:
        print(f"🔔 ¡ALERTA ACTIVADA! {alert['crypto']}: {current_value} {condition} {threshold}")
        return send_alert_notification(alert, current_value)
    else:
        print(f"✓ No activada ({current_value} vs {threshold})")
        return False


def check_smart_alert(alert: dict) -> bool:
    """
    Verifica alertas inteligentes (mínimo/máximo mensual, cambios 24h)
    
    Args:
        alert: Diccionario con configuración
        
    Returns:
        True si se activó
    """
    if not alert.get('enabled', False):
        print(f"⏭️  Alerta deshabilitada: {alert.get('crypto')}")
        return False
    
    crypto = alert['crypto']
    alert_type = alert['type']
    
    print(f"\n🔍 Revisando alerta inteligente: {crypto} - {alert_type}")
    
    if alert_type == 'minimo_mensual':
        monthly_data = get_monthly_low(crypto, alert.get('exchange', 'kraken'))
        
        if monthly_data and monthly_data['is_below']:
            print(f"🔴 ¡SOPORTE ROTO! {crypto} cayó bajo mínimo mensual")
            return send_support_break_alert(crypto, monthly_data)
        else:
            print(f"✓ Soporte intacto")
    
    elif alert_type == 'maximo_mensual':
        monthly_data = get_monthly_high(crypto, alert.get('exchange', 'kraken'))
        
        if monthly_data and monthly_data['is_above']:
            print(f"🟢 ¡RESISTENCIA ROTA! {crypto} superó máximo mensual")
            return send_resistance_break_alert(crypto, monthly_data)
        else:
            print(f"✓ Resistencia intacta")
    
    elif alert_type == 'cambio_24h':
        change_data = calculate_24h_change(crypto, alert.get('exchange', 'kraken'))
        
        if change_data:
            change_pct = change_data['change_pct']
            threshold = alert['threshold']
            
            if abs(change_pct) > threshold:
                emoji = "🚀" if change_pct > 0 else "💥"
                direction = "subió" if change_pct > 0 else "cayó"
                
                message = f"""
{emoji} *CAMBIO EXTREMO 24H - ALERTA CRÍTICA*

💰 *{crypto}*

📊 *Cambio en 24h:* {change_pct:+.2f}%
💵 Diferencia: ${change_data['change_abs']:+,.2f}

📉 Hace 24h: ${change_data['price_24h_ago']:,.2f}
📈 Ahora: ${change_data['current']:,.2f}

⚠️ *Volatilidad extrema detectada*

El precio {direction} {abs(change_pct):.1f}% en las últimas 24 horas.

🎯 *Implicaciones:*
• Alta volatilidad
• Posible continuación del movimiento
• Revisar volumen y noticias

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

_CryptoView Pro by Julian E. Coronado Gil_
"""
                send_telegram(message)
                print(f"🔔 ¡ALERTA ACTIVADA! Cambio extremo: {change_pct:+.2f}%")
                return True
            else:
                print(f"✓ Cambio normal: {change_pct:+.2f}% (umbral: {threshold}%)")
    
    return False


# ============ FUNCIÓN PRINCIPAL ============

def main():
    """
    Función principal que ejecuta la revisión de alertas
    """
    print("=" * 60)
    print("🔔 CRYPTOVIEW PRO - SISTEMA DE ALERTAS INTELIGENTES")
    print("=" * 60)
    print(f"⏰ Ejecutado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👨‍💻 Developed by Julian E. Coronado Gil")
    print("=" * 60)
    
    # Verificar configuración de Telegram
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n❌ ERROR: Variables de entorno no configuradas")
        print("Configura TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en GitHub Secrets")
        return
    
    print(f"\n✅ Telegram configurado")
    print(f"📱 Chat ID: {TELEGRAM_CHAT_ID}")
    
    # Cargar alertas
    alerts = load_alerts()
    
    if not alerts:
        print("\n⚠️ No hay alertas configuradas")
        print(f"Crea alertas en: {ALERTS_FILE}")
        
        send_telegram(
            "⚠️ *Sistema de Alertas Activo*\n\n"
            "No hay alertas configuradas.\n"
            f"Configuradas: 0\n\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return
    
    print(f"\n📋 Alertas encontradas: {len(alerts)}")
    
    # Revisar cada alerta
    triggered_count = 0
    enabled_count = sum(1 for a in alerts if a.get('enabled', False))
    
    for i, alert in enumerate(alerts, 1):
        print(f"\n{'=' * 50}")
        print(f"Alerta {i}/{len(alerts)}")
        print(f"{'=' * 50}")
        
        # Detectar tipo de alerta
        alert_type = alert.get('type')
        
        if alert_type in ['minimo_mensual', 'maximo_mensual', 'cambio_24h']:
            # Alertas inteligentes
            if check_smart_alert(alert):
                triggered_count += 1
        elif alert_type in ['precio', 'rsi']:
            # Alertas normales
            if check_alert(alert):
                triggered_count += 1
        else:
            print(f"⚠️ Tipo de alerta desconocido: {alert_type}")
    
    # Resumen
    print("\n" + "=" * 60)
    print(f"✅ REVISIÓN COMPLETADA")
    print("=" * 60)
    print(f"📊 Alertas habilitadas: {enabled_count}/{len(alerts)}")
    print(f"🔔 Alertas activadas: {triggered_count}")
    print(f"⏰ Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Enviar resumen si no se activó ninguna
    if triggered_count == 0 and enabled_count > 0:
        send_telegram(
            f"✅ *Sistema de Alertas - Todo en Orden*\n\n"
            f"📊 Alertas monitoreadas: {enabled_count}\n"
            f"🔔 Alertas activadas: 0\n\n"
            f"Todos los niveles bajo control 👍\n\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        
        # Enviar error a Telegram
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            send_telegram(
                f"🚨 *ERROR en Sistema de Alertas*\n\n"
                f"```\n{str(e)}\n```\n\n"
                f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
