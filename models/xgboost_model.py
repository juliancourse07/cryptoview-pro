"""
Modelo XGBoost para predicción de criptomonedas
Optimizado para series temporales financieras
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class XGBoostCryptoPredictor:
    """
    Predictor de criptomonedas usando XGBoost con ingeniería de features avanzada
    """
    
    def __init__(self, 
                 n_estimators: int = 200,
                 learning_rate: float = 0.07,
                 max_depth: int = 6,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8):
        """
        Inicializa el modelo XGBoost
        
        Args:
            n_estimators: Número de árboles
            learning_rate: Tasa de aprendizaje
            max_depth: Profundidad máxima de árboles
            subsample: Fracción de muestras para cada árbol
            colsample_bytree: Fracción de features para cada árbol
        """
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        self.trained = False
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features de ingeniería para el modelo
        
        Features creados:
        - Retornos (1h, 4h, 24h, 7d)
        - Medias móviles (7, 14, 30 períodos)
        - Volatilidad (ventanas móviles)
        - RSI, MACD (si existen)
        - Features temporales (hora, día de semana)
        - Ratios de precio
        """
        df = df.copy()
        
        # 1. RETORNOS EN DIFERENTES VENTANAS
        df['return_1'] = df['close'].pct_change(1)
        df['return_4'] = df['close'].pct_change(4)
        df['return_24'] = df['close'].pct_change(24)
        df['return_168'] = df['close'].pct_change(168)  # 7 días
        
        # 2. MEDIAS MÓVILES
        for window in [7, 14, 30, 50]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'price_to_ma_{window}'] = df['close'] / df[f'ma_{window}']
        
        # 3. MEDIAS MÓVILES EXPONENCIALES
        for span in [12, 26, 50]:
            df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        
        # 4. VOLATILIDAD
        for window in [7, 14, 30]:
            df[f'volatility_{window}'] = df['return_1'].rolling(window=window).std()
        
        # 5. MOMENTUM
        df['momentum_7'] = df['close'] - df['close'].shift(7)
        df['momentum_14'] = df['close'] - df['close'].shift(14)
        
        # 6. BANDAS DE BOLLINGER
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 7. FEATURES DE VOLUMEN
        if 'volume' in df.columns:
            df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_7']
            df['volume_change'] = df['volume'].pct_change(1)
        
        # 8. RATIOS OHLC
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # 9. FEATURES TEMPORALES (si el índice es datetime)
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
        
        # 10. RSI y MACD si ya existen
        if 'rsi' in df.columns:
            df['rsi_normalized'] = df['rsi'] / 100
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_diff'] = df['macd'] - df['macd_signal']
            df['macd_positive'] = (df['macd_diff'] > 0).astype(int)
        
        # 11. LAGS (valores pasados)
        for lag in [1, 2, 3, 7, 14]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
        
        # 12. TARGET: Precio futuro (1 período adelante)
        df['target'] = df['close'].shift(-1)
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, train_size: float = 0.8) -> Tuple:
        """
        Prepara datos para entrenamiento
        
        Args:
            df: DataFrame con datos históricos
            train_size: Fracción de datos para entrenamiento (0.0-1.0)
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Crear features
        df_features = self.create_features(df)
        
        # Eliminar filas con NaN
        df_features = df_features.dropna()
        
        if len(df_features) < 100:
            raise ValueError("No hay suficientes datos después de crear features (mínimo 100)")
        
        # Separar features y target
        feature_cols = [col for col in df_features.columns 
                       if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
        
        X = df_features[feature_cols]
        y = df_features['target']
        
        self.feature_columns = feature_cols
        
        # Split temporal (no random, respetamos el orden del tiempo)
        split_idx = int(len(X) * train_size)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
    
    def train(self, df: pd.DataFrame, train_size: float = 0.8) -> Dict:
        """
        Entrena el modelo
        
        Args:
            df: DataFrame con datos históricos
            train_size: Fracción de datos para entrenamiento
            
        Returns:
            Diccionario con métricas de entrenamiento
        """
        try:
            # Preparar datos
            X_train, X_test, y_train, y_test = self.prepare_data(df, train_size)
            
            # Entrenar modelo
            self.model.fit(
                X_train, 
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            self.trained = True
            
            # Predicciones
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            
            # Métricas
            metrics = {
                'train_mae': np.mean(np.abs(y_train - train_pred)),
                'test_mae': np.mean(np.abs(y_test - test_pred)),
                'train_rmse': np.sqrt(np.mean((y_train - train_pred) ** 2)),
                'test_rmse': np.sqrt(np.mean((y_test - test_pred) ** 2)),
                'train_mape': np.mean(np.abs((y_train - train_pred) / y_train)) * 100,
                'test_mape': np.mean(np.abs((y_test - test_pred) / y_test)) * 100,
            }
            
            # Precisión de dirección
            train_direction = np.sign(np.diff(y_train))
            train_pred_direction = np.sign(np.diff(train_pred))
            metrics['train_direction_accuracy'] = np.mean(train_direction == train_pred_direction) * 100
            
            test_direction = np.sign(np.diff(y_test))
            test_pred_direction = np.sign(np.diff(test_pred))
            metrics['test_direction_accuracy'] = np.mean(test_direction == test_pred_direction) * 100
            
            return metrics
            
        except Exception as e:
            raise Exception(f"Error entrenando modelo: {str(e)}")
    
    def predict_future(self, df: pd.DataFrame, periods: int = 24) -> pd.DataFrame:
        """
        Predice precios futuros
        
        Args:
            df: DataFrame con datos históricos
            periods: Número de períodos a predecir
            
        Returns:
            DataFrame con predicciones
        """
        if not self.trained:
            raise ValueError("El modelo debe ser entrenado primero. Usa train() antes de predecir.")
        
        predictions = []
        df_work = df.copy()
        
        for i in range(periods):
            # Crear features para el último punto
            df_features = self.create_features(df_work)
            df_features = df_features.dropna()
            
            if len(df_features) == 0:
                break
            
            # Obtener última fila
            last_row = df_features[self.feature_columns].iloc[-1:].values
            
            # Escalar
            last_row_scaled = self.scaler.transform(last_row)
            
            # Predecir
            pred = self.model.predict(last_row_scaled)[0]
            predictions.append(pred)
            
            # Crear nueva fila para siguiente predicción
            last_datetime = df_work.index[-1]
            
            # Detectar frecuencia
            if len(df_work) >= 2:
                freq = df_work.index[-1] - df_work.index[-2]
            else:
                freq = pd.Timedelta(hours=1)
            
            next_datetime = last_datetime + freq
            
            # Nueva fila con predicción
            new_row = pd.DataFrame({
                'open': [pred],
                'high': [pred * 1.01],
                'low': [pred * 0.99],
                'close': [pred],
                'volume': [df_work['volume'].iloc[-1] if 'volume' in df_work.columns else 0]
            }, index=[next_datetime])
            
            df_work = pd.concat([df_work, new_row])
        
        # Crear DataFrame de predicciones
        last_datetime = df.index[-1]
        
        if len(df) >= 2:
            freq = df.index[-1] - df.index[-2]
        else:
            freq = pd.Timedelta(hours=1)
        
        future_dates = [last_datetime + freq * (i + 1) for i in range(len(predictions))]
        
        predictions_df = pd.DataFrame({
            'timestamp': future_dates,
            'predicted_price': predictions
        })
        predictions_df.set_index('timestamp', inplace=True)
        
        return predictions_df
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Obtiene importancia de features
        
        Returns:
            DataFrame con features ordenados por importancia
        """
        if not self.trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df


# ============ FUNCIONES DE UTILIDAD ============

def backtest_model(df: pd.DataFrame, 
                   predictor: XGBoostCryptoPredictor,
                   train_size: float = 0.8) -> Dict:
    """
    Realiza backtesting del modelo
    
    Args:
        df: DataFrame con datos históricos
        predictor: Instancia de XGBoostCryptoPredictor
        train_size: Fracción de datos para entrenamiento
        
    Returns:
        Diccionario con resultados del backtesting
    """
    # Entrenar
    metrics = predictor.train(df, train_size)
    
    # Preparar datos para comparación
    X_train, X_test, y_train, y_test = predictor.prepare_data(df, train_size)
    
    # Predicciones
    train_pred = predictor.model.predict(X_train)
    test_pred = predictor.model.predict(X_test)
    
    results = {
        'metrics': metrics,
        'train_actual': y_train,
        'train_predicted': train_pred,
        'test_actual': y_test,
        'test_predicted': test_pred,
        'feature_importance': predictor.get_feature_importance()
    }
    
    return results


def create_prediction_intervals(predictions: pd.DataFrame, 
                                confidence: float = 0.95) -> pd.DataFrame:
    """
    Crea intervalos de confianza para predicciones
    
    Args:
        predictions: DataFrame con predicciones
        confidence: Nivel de confianza (0.0-1.0)
        
    Returns:
        DataFrame con intervalos superior e inferior
    """
    predictions = predictions.copy()
    
    # Estimar variabilidad (método simple)
    # En producción, usarías quantile regression
    std_estimate = predictions['predicted_price'].std()
    
    # Z-score para intervalo de confianza
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    margin = z_score * std_estimate
    
    predictions['lower_bound'] = predictions['predicted_price'] - margin
    predictions['upper_bound'] = predictions['predicted_price'] + margin
    
    return predictions
