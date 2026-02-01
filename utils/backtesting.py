"""
Motor de backtesting mejorado
"""
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Backtester:
    """Ejecuta backtesting de modelos con métricas avanzadas"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """
        Calcula métricas completas de evaluación
        
        Args:
            actual: Valores reales
            predicted: Valores predichos
            
        Returns:
            Diccionario con todas las métricas
        """
        # Métricas de error
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # R² Score
        r2 = r2_score(actual, predicted)
        
        # Precisión de dirección
        actual_direction = np.sign(np.diff(actual))
        pred_direction = np.sign(np.diff(predicted))
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # Error máximo
        max_error = np.max(np.abs(actual - predicted))
        
        # Métricas adicionales
        relative_error = np.abs((actual - predicted) / actual)
        median_ape = np.median(relative_error) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Median_APE': median_ape,
            'R2_Score': r2,
            'Direction_Accuracy': direction_accuracy,
            'Max_Error': max_error,
            'Mean_Actual': np.mean(actual),
            'Mean_Predicted': np.mean(predicted),
            'Std_Actual': np.std(actual),
            'Std_Predicted': np.std(predicted)
        }
    
    def format_metrics(self, metrics: Dict) -> str:
        """
        Formatea métricas para display
        
        Args:
            metrics: Diccionario de métricas
            
        Returns:
            String formateado
        """
        formatted = []
        formatted.append("### 📊 Métricas de Evaluación\n")
        formatted.append(f"**MAE (Error Absoluto Medio):** ${metrics['MAE']:.2f}")
        formatted.append(f"**RMSE (Raíz del Error Cuadrático):** ${metrics['RMSE']:.2f}")
        formatted.append(f"**MAPE (Error Porcentual Medio):** {metrics['MAPE']:.2f}%")
        formatted.append(f"**Precisión de Dirección:** {metrics['Direction_Accuracy']:.2f}%")
        formatted.append(f"**R² Score:** {metrics['R2_Score']:.4f}")
        
        return "\n\n".join(formatted)
