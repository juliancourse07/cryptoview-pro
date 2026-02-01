"""
Indicadores técnicos para análisis
"""
import pandas as pd
import numpy as np
from typing import Tuple

class TechnicalIndicators:
    """Calcula indicadores técnicos comunes"""
    
    @staticmethod
    def calculate_rsi(data: pd.Series, periods: int = 14) -> pd.Series:
        """Relative Strength Index (RSI)"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Moving Average Convergence Divergence (MACD)"""
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Agrega todos los indicadores al DataFrame"""
        df = df.copy()
        
        # RSI
        df['rsi'] = TechnicalIndicators.calculate_rsi(df['close'])
        
        # MACD
        macd, signal, hist = TechnicalIndicators.calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # Bollinger Bands
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(df['close'])
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        # EMAs
        for period in [9, 21, 50, 200]:
            df[f'ema_{period}'] = TechnicalIndicators.calculate_ema(df['close'], period)
        
        return df
    
    @staticmethod
    def get_signals(df: pd.DataFrame) -> dict:
        """Genera señales de trading basadas en indicadores"""
        signals = {
            'rsi_signal': 'neutral',
            'macd_signal': 'neutral',
            'bb_signal': 'neutral',
            'overall': 'neutral'
        }
        
        if df.empty or 'rsi' not in df.columns:
            return signals
        
        last_row = df.iloc[-1]
        
        # RSI Signal
        if last_row['rsi'] > 70:
            signals['rsi_signal'] = 'overbought'
        elif last_row['rsi'] < 30:
            signals['rsi_signal'] = 'oversold'
        
        # MACD Signal
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            if last_row['macd'] > last_row['macd_signal']:
                signals['macd_signal'] = 'bullish'
            else:
                signals['macd_signal'] = 'bearish'
        
        # Overall Signal
        bullish_count = sum([
            signals['rsi_signal'] == 'oversold',
            signals['macd_signal'] == 'bullish'
        ])
        
        bearish_count = sum([
            signals['rsi_signal'] == 'overbought',
            signals['macd_signal'] == 'bearish'
        ])
        
        if bullish_count >= 2:
            signals['overall'] = 'buy'
        elif bearish_count >= 2:
            signals['overall'] = 'sell'
        
        return signals
