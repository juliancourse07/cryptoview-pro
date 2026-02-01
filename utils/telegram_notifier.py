"""
Sistema de notificaciones por Telegram
"""
import requests
from typing import Optional
from datetime import datetime

class TelegramNotifier:
    """
    Envía notificaciones a Telegram
    """
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Inicializa el notificador
        
        Args:
            bot_token: Token del bot de Telegram
            chat_id: ID del chat donde enviar mensajes
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send_message(self, message: str, parse_mode: str = 'Markdown') -> bool:
        """
        Envía un mensaje de texto
        
        Args:
            message: Texto del mensaje
            parse_mode: 'Markdown' o 'HTML'
            
        Returns:
            True si se envió exitosamente
        """
        try:
            url = f"{self.api_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            
            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            print(f"Error enviando mensaje: {e}")
            return False
    
    def send_alert(self, 
                   crypto: str,
                   alert_type: str,
                   current_value: float,
                   threshold: float,
                   condition: str) -> bool:
        """
        Envía una alerta formateada
        
        Args:
            crypto: Símbolo de la cripto
            alert_type: Tipo de alerta (Precio, RSI, etc)
            current_value: Valor actual
            threshold: Umbral configurado
            condition: Condición (Mayor que, Menor que)
            
        Returns:
            True si se envió
        """
        message = f"""
🚨 *ALERTA CRYPTOVIEW PRO*

💰 *{crypto}*
📊 Tipo: {alert_type}

🎯 Condición: {condition} {threshold}
📈 Valor actual: {current_value:.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

_Configura tus alertas en CryptoView Pro_
"""
        return self.send_message(message)
    
    def send_prediction_alert(self,
                             crypto: str,
                             current_price: float,
                             predicted_price: float,
                             hours: int,
                             change_pct: float,
                             confidence: float) -> bool:
        """
        Envía alerta de predicción ML
        
        Args:
            crypto: Símbolo
            current_price: Precio actual
            predicted_price: Precio predicho
            hours: Horizonte en horas
            change_pct: Cambio porcentual
            confidence: Nivel de confianza
            
        Returns:
            True si se envió
        """
        emoji = "📈" if change_pct > 0 else "📉"
        
        message = f"""
🔮 *PREDICCIÓN ML - CRYPTOVIEW PRO*

💰 *{crypto}*

📊 *Precio Actual:* ${current_price:,.2f}
{emoji} *Predicción ({hours}h):* ${predicted_price:,.2f}
📈 *Cambio Esperado:* {change_pct:+.2f}%
🎯 *Confianza:* {confidence:.1f}%

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

_Generado por modelos híbridos XGBoost + Prophet_
"""
        return self.send_message(message)
    
    def send_signal(self,
                   crypto: str,
                   signal: str,
                   current_price: float,
                   rsi: float,
                   macd_signal: str) -> bool:
        """
        Envía señal de trading
        
        Args:
            crypto: Símbolo
            signal: COMPRA, VENTA, NEUTRAL
            current_price: Precio actual
            rsi: Valor RSI
            macd_signal: Señal MACD
            
        Returns:
            True si se envió
        """
        emoji = "🟢" if signal == "COMPRA" else "🔴" if signal == "VENTA" else "🟡"
        
        message = f"""
{emoji} *SEÑAL DE TRADING*

💰 *{crypto}*
💵 Precio: ${current_price:,.2f}

🎲 *Señal: {signal}*

📊 Indicadores:
• RSI: {rsi:.1f}
• MACD: {macd_signal}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️ _No es asesoría financiera. Investiga antes de operar._
"""
        return self.send_message(message)
    
    def test_connection(self) -> bool:
        """
        Prueba la conexión enviando un mensaje de test
        
        Returns:
            True si funciona
        """
        message = "✅ *Conexión exitosa con CryptoView Pro*\n\n¡Tu bot de alertas está activo!"
        return self.send_message(message)


def create_telegram_bot_url(bot_token: str) -> str:
    """
    Genera URL para abrir el bot en Telegram
    
    Args:
        bot_token: Token del bot
        
    Returns:
        URL de Telegram
    """
    bot_username = bot_token.split(':')[0]
    return f"https://t.me/{bot_username}"
