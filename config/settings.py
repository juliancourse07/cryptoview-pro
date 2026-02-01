"""
Configuración global del sistema
"""
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

# ============ API CONFIGURATION ============
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')

# ============ CRYPTO SETTINGS ============
AVAILABLE_CRYPTOS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 
    'ADA/USDT', 'XRP/USDT', 'DOGE/USDT', 'DOT/USDT',
    'MATIC/USDT', 'AVAX/USDT', 'LINK/USDT', 'UNI/USDT'
]

TIMEFRAMES = {
    '1m': '1 minuto',
    '5m': '5 minutos',
    '15m': '15 minutos',
    '1h': '1 hora',
    '4h': '4 horas',
    '1d': '1 día'
}

DEFAULT_CRYPTO = os.getenv('DEFAULT_CRYPTO', 'BTC/USDT')
DEFAULT_TIMEFRAME = os.getenv('DEFAULT_TIMEFRAME', '1h')

# ============ MODEL SETTINGS ============
MODELS_CONFIG = {
    'prophet': {
        'changepoint_prior_scale': 0.5,
        'seasonality_prior_scale': 10,
        'seasonality_mode': 'multiplicative',
        'daily_seasonality': True,
        'weekly_seasonality': True,
        'yearly_seasonality': False
    },
    'lstm': {
        'sequence_length': 60,
        'epochs': 50,
        'batch_size': 32,
        'units': [128, 64, 32],
        'dropout': 0.2
    },
    'xgboost': {
        'n_estimators': 200,
        'learning_rate': 0.07,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
}

# ============ FORECAST SETTINGS ============
FORECAST_HOURS = int(os.getenv('FORECAST_HOURS', 24))
AUTO_REFRESH_SECONDS = int(os.getenv('AUTO_REFRESH_SECONDS', 30))
CACHE_TTL = 60
DATA_LIMIT = 1000
MIN_DATA_POINTS = 100

# ============ TECHNICAL INDICATORS ============
INDICATORS_CONFIG = {
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'bb_std': 2,
    'ema_periods': [9, 21, 50, 200]
}

# ============ ALERTS SETTINGS ============
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

ALERT_THRESHOLDS = {
    'price_change_pct': 5.0,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'volume_spike': 2.0
}

# ============ STYLING ============
COLORS = {
    'background': '#0e1117',
    'text': '#fafafa',
    'accent': '#00d9ff',
    'success': '#00ff88',
    'warning': '#ffaa00',
    'danger': '#ff4444',
    'forecast': '#ffd700'
}

# CSS SIMPLIFICADO Y SEGURO
CUSTOM_CSS = """
<style>
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    
    .big-metric {
        font-size: 32px;
        font-weight: 800;
        color: #00d9ff;
    }
    
    .positive {
        color: #00ff88 !important;
    }
    
    .negative {
        color: #ff4444 !important;
    }
    
    .neutral {
        color: #ffaa00 !important;
    }
    
    /* Mejoras visuales sutiles */
    .stMetric {
        background: rgba(255, 255, 255, 0.02);
        padding: 10px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 8px 16px;
        margin: 0 4px;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: rgba(0, 217, 255, 0.2);
    }
</style>
"""
