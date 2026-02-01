"""
Modelo Híbrido: Combina XGBoost (corto plazo) + Prophet (largo plazo)
Mejor de ambos mundos
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple

try:
    from models.xgboost_model import XGBoostCryptoPredictor
    from models.prophet_model import ProphetCryptoPredictor
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

class HybridCryptoPredictor:
    """
    Modelo híbrido que usa:
    - XGBoost para predicciones de corto plazo (1-72 horas)
    - Prophet para predicciones de mediano/largo plazo (1 semana - 1 mes)
    """
    
    def __init__(self):
        if not MODELS_AVAILABLE:
            raise ImportError("Modelos no disponibles")
        
        self.xgboost = XGBoostCryptoPredictor()
        self.prophet = ProphetCryptoPredictor()
        self.trained = False
        self.training_info = {}
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Entrena ambos modelos"""
        print("🎯 Entrenando XGBoost (corto plazo)...")
        xgb_metrics = self.xgboost.train(df, train_size=0.8)
        
        print("📊 Entrenando Prophet (largo plazo)...")
        prophet_metrics = self.prophet.train(df)
        
        self.trained = True
        self.training_info = {
            'xgboost': xgb_metrics,
            'prophet': prophet_metrics,
            'data_points': len(df)
        }
        
        return self.training_info
    
    def predict_future(self, df: pd.DataFrame, periods: int) -> Dict[str, pd.DataFrame]:
        """
        Predice usando el modelo apropiado según horizonte
        
        Args:
            df: DataFrame histórico
            periods: Horas a predecir
            
        Returns:
            Dict con predicciones de ambos modelos
        """
        if not self.trained:
            raise ValueError("Modelos deben ser entrenados primero")
        
        predictions = {}
        
        # XGBoost para corto plazo (siempre cuando periods <= 72)
        if periods <= 72:
            predictions['xgboost'] = self.xgboost.predict_future(df, periods=periods)
            from models.xgboost_model import create_prediction_intervals
            predictions['xgboost'] = create_prediction_intervals(predictions['xgboost'])
            predictions['recommended'] = 'xgboost'
        
        # Prophet para mediano/largo plazo
        if periods > 24:
            predictions['prophet'] = self.prophet.predict_future(periods=periods, freq='H')
            if periods > 72:
                predictions['recommended'] = 'prophet'
        
        # Ambos disponibles, crear ensemble
        if 'xgboost' in predictions and 'prophet' in predictions:
            xgb_weight = max(0, 1 - (periods / 168))
            prophet_weight = 1 - xgb_weight
            
            combined = predictions['prophet'].copy()
            xgb_pred = predictions['xgboost']['predicted_price']
            prophet_pred = predictions['prophet']['predicted_price']
            
            overlap_len = min(len(xgb_pred), len(prophet_pred))
            combined.loc[combined.index[:overlap_len], 'predicted_price'] = (
                xgb_pred[:overlap_len].values * xgb_weight +
                prophet_pred[:overlap_len].values * prophet_weight
            )
            
            # Promediar bounds también
            if 'lower_bound' in predictions['xgboost'].columns:
                combined.loc[combined.index[:overlap_len], 'lower_bound'] = (
                    predictions['xgboost']['lower_bound'][:overlap_len].values * xgb_weight +
                    predictions['prophet']['lower_bound'][:overlap_len].values * prophet_weight
                )
                combined.loc[combined.index[:overlap_len], 'upper_bound'] = (
                    predictions['xgboost']['upper_bound'][:overlap_len].values * xgb_weight +
                    predictions['prophet']['upper_bound'][:overlap_len].values * prophet_weight
                )
            
            predictions['hybrid'] = combined
            predictions['weights'] = {'xgboost': xgb_weight, 'prophet': prophet_weight}
            predictions['recommended'] = 'hybrid'
        
        return predictions
    
    def get_best_prediction(self, predictions: Dict) -> pd.DataFrame:
        """
        Retorna la mejor predicción según el horizonte
        """
        if 'hybrid' in predictions:
            return predictions['hybrid']
        elif predictions.get('recommended') == 'xgboost':
            return predictions['xgboost']
        else:
            return predictions['prophet']
    
    def get_training_info(self) -> Dict:
        """Retorna información del entrenamiento"""
        return self.training_info
