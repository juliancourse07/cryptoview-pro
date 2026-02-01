"""
🔔 Sistema Automático de Alertas para CryptoView Pro
Ejecutado por GitHub Actions cada hora

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
    Envía notificación formateada de alerta
    
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
    Verifica si se cumple una alerta
    
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


# ============ FUNCIÓN PRINCIPAL ============

def main():
    """
    Función principal que ejecuta la revisión de alertas
    """
    print("=" * 60)
    print("🔔 CRYPTOVIEW PRO - SISTEMA DE ALERTAS AUTOMÁTICO")
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
        
        # Enviar notificación de que no hay alertas
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
        print(f"\n--- Alerta {i}/{len(alerts)} ---")
        if check_alert(alert):
            triggered_count += 1
    
    # Resumen
    print("\n" + "=" * 60)
    print(f"✅ Revisión completada")
    print(f"📊 Alertas revisadas: {enabled_count}/{len(alerts)}")
    print(f"🔔 Alertas activadas: {triggered_count}")
    print("=" * 60)
    
    # Enviar resumen si no se activó ninguna (opcional)
    if triggered_count == 0 and enabled_count > 0:
        send_telegram(
            f"✅ *Sistema de Alertas - Revisión Completada*\n\n"
            f"📊 Alertas activas: {enabled_count}\n"
            f"🔔 Alertas activadas: 0\n\n"
            f"Todo en orden 👍\n\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        
        # Enviar error a Telegram
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            send_telegram(
                f"🚨 *ERROR en Sistema de Alertas*\n\n"
                f"```\n{str(e)}\n```\n\n"
                f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
